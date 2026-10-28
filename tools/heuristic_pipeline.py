#!/usr/bin/env python3
"""Possession from four simple heuristics, plus a visual report that explains every decision.

A teaching pipeline, not a competition entry. Each step answers one question with specific data,
fixes one kind of error, and hands what it cannot decide ("-", no answer) to the next step:

    1. NEAREST PLAYER   who is near the ball?          ATD ball + track slots + slot teams (XML)
    2. DEAD BALL        is the ball in play?           ball off the pitch, ball and player speeds,
                                                       Smart Tagging set pieces
    3. CONTROL          control, or just a touch?      step 2 over time (runs shorter than 1 s)
    4. GAPS             what happens without a ball?   step 3 + which team starts advancing

Usage (from the repository root):

    python tools/heuristic_pipeline.py --data-dir fake_data/seed0
    python tools/heuristic_pipeline.py --data-dir fake_data/seed0 --window-half 2 --window-minute 10

It writes, next to the data it read (so a report built from real data never lands in the repo):

    <data-dir>/reports/heuristics/report.html        open it in a browser
    <data-dir>/reports/heuristics/submissions/*.csv  one valid submission per step + the baseline
    <data-dir>/reports/heuristics/summary.json       aggregate numbers only

Ground truth, if any:
    - fake data: <data-dir>/BAR-ATH_possession_gt_fake.csv is used automatically (full match);
    - real data: only the public 5 minutes, shown as a PIPELINE CHECK. Never tune on it (RULES.md).

REAL DATA: the report embeds a few minutes of tracking for the explorer, so it is match data.
Keep it on your machine, never commit it and never show it to a cloud LLM (Claude included).
The console prints aggregates only. Every parameter below is a starting guess, not a tuned value.

Dependencies: numpy and pandas, like the official scorer. The report is one self-contained HTML
file with no external resources.
"""

import argparse
import datetime
import io
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import load_smart_tagging  # noqa: E402
from baseline import smooth as baseline_smooth, team_of_each_track  # noqa: E402

GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.source_frames("tactical", GAME)
L, W = games.GAMES[GAME]["pitch"]
HALVES = games.half_video_blocks(GAME)          # [(1, 11185, 90651), (2, 110808, 190197)]
TEAMS = "HA"
NONE = "-"                                       # "no answer yet": the next step decides

DEFAULTS = {
    "radius_m": 2.0,         # step 1: a player this close to the ball has it
    "min_control_s": 1.0,    # step 3: a team run shorter than this is a touch, not control
    "fast_ball_ms": 4.0,     # step 2: the ball moving faster than this is in play
    "max_dead_s": 60.0,      # step 2: longest stoppage we will draw before a restart
    "out_margin_m": 0.5,     # step 2: how far past a line the ball must be to count as out
    "out_min_frames": 5,     # step 2: and for how many frames in a row
    "still_speed_ms": 1.5,   # step 2: players walking (median speed below this) ...
    "still_min_s": 4.0,      # step 2: ... for at least this long, with no fast ball = play stopped
    "min_split_gap_s": 0.5,  # step 4: gaps shorter than this are split in the middle
}

STEP_NAMES = ["Paso 1 · más cercano", "Paso 2 · balón muerto", "Paso 3 · control", "Paso 4 · huecos"]
STEP_FILES = ["step1_nearest", "step2_dead_ball", "step3_control", "step4_gaps"]


# ==================================================================================================
# Loading
# ==================================================================================================

def load_tracking(paths, cache_path):
    """(players[n, slots, 2], ball[n, 2], slot_team[slots]) in metres, indexed by video frame.

    Reads the raw ATD with pandas instead of kloppy: one pass, a few seconds, and robust to the
    number of channels per slot (x, y, speed -- or x, y, mapping id, speed). Speeds in the file
    are ignored; we compute our own from positions.
    """
    slot_team = team_of_each_track(paths["atd_meta"]).astype(np.int8)
    raw = paths["atd_raw"]
    if cache_path and os.path.isfile(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(raw):
        cached = np.load(cache_path)
        return cached["players"], cached["ball"], slot_team

    with open(raw) as handle:
        first = handle.readline().rstrip("\r\n").split(":")
    groups = first[1].split(";")
    n_slots, per_slot = len(groups), len(groups[0].split(","))
    if n_slots != len(slot_team):
        raise ValueError("ATD has %d slots, metadata lists %d" % (n_slots, len(slot_team)))
    ball_col = 1 + n_slots * per_slot
    usecols = [0] + [1 + k * per_slot + c for k in range(n_slots) for c in (0, 1)] + [ball_col, ball_col + 1]

    with open(raw) as handle:
        text = handle.read().translate(str.maketrans(":;", ",,"))
    table = pd.read_csv(io.StringIO(text), header=None, usecols=usecols, dtype=np.float64,
                        na_values=["NaN", "nan"], engine="c").to_numpy()
    del text

    video = table[:, 0].astype(np.int64) - 1                     # ATD frame_id counts from 1
    players = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    xy = table[:, 1:1 + 2 * n_slots].reshape(-1, n_slots, 2)
    players[video] = xy * np.array([L, W])
    ball[video] = table[:, -2:] * np.array([L, W])
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, players=players, ball=ball)
    return players, ball, slot_team


def load_restarts(data_dir, params):
    """Set pieces from the Smart Tagging: [{frame_lo, frame_hi, type, team}]. Restart refined later."""
    tag_team = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}
    out = []
    for event in load_smart_tagging.load(GAME, data_dir):
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        out.append({"lo": event["start_frame"], "hi": event["end_frame"],
                    "type": (event["tags"].get("Type") or "?").strip(),
                    "team": tag_team.get((event["tags"].get("Team") or "").strip().lower(), "?")})
    out.sort(key=lambda r: r["lo"])
    return out


def load_truth(data_dir):
    """(labels as an array of 'H'/'A'/'D'/'X', kind) -- kind is 'full', 'public' or None."""
    fake = os.path.join(data_dir, "%s_possession_gt_fake.csv" % GAME)
    if os.path.isfile(fake):
        return pm.read_label_csv(fake)["label"].astype(str).str.strip().str.upper().to_numpy(), "full"
    public = os.path.join(ROOT, "evaluation", "gt", "%s_possession_gt_public.csv" % GAME)
    if os.path.isfile(public):
        return pm.read_label_csv(public)["label"].astype(str).str.strip().str.upper().to_numpy(), "public"
    return None, None


# ==================================================================================================
# Small helpers
# ==================================================================================================

def graded_mask():
    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in HALVES:
        mask[first:last + 1] = True
    return mask


def half_of(frame):
    for half, first, last in HALVES:
        if first <= frame <= last:
            return half, first, last
    return None, None, None


def speed_of(xy, k=2):
    """Speed in m/s over +/-k frames. Works for (n, 2) and (n, slots, 2). NaN where unknown."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) * FPS / (2 * k)
    return out


def runs_of(values):
    """[(start, stop, value)] maximal constant runs of a 1-D array."""
    if not len(values):
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def live_stretches(labels):
    """[(start, stop)] maximal non-D spans inside each graded half."""
    out = []
    for _, first, last in HALVES:
        for a, b, v in runs_of(labels[first:last + 1] == "D"):
            if not v:
                out.append((first + a, first + b))
    return out


def clock(frame):
    """'1ª 12:34.5' -- time since the start of the graded half."""
    half, first, _ = half_of(frame)
    if half is None:
        return "fuera de juego"
    tenths = int(round((frame - first) * 10.0 / FPS))
    return "%dª %02d:%02d.%d" % (half, tenths // 600, tenths // 10 % 60, tenths % 10)


def for_scoring(labels):
    """A valid submission: '-' becomes X (abstain) and everything outside the halves is X."""
    out = np.where(labels == NONE, "X", labels).astype("<U1")
    out[~graded_mask()] = "X"
    return out


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)


# ==================================================================================================
# Step 1 -- nearest player
# ==================================================================================================

def step1_nearest(players, ball, slot_team, params):
    """Label each frame with the team of the nearest player, if that player is close enough.

    reason: 0 ball not seen, 1 a player within radius, 2 ball seen but nobody close (in the air?).
    Also returns the official baseline's per-frame codes (nearest at any distance), so we can
    rebuild baseline_smooth50 from the same arrays.
    """
    labels = np.full(N_FRAMES, NONE, dtype="<U1")
    reason = np.zeros(N_FRAMES, dtype=np.int8)
    dist = np.full(N_FRAMES, np.nan, dtype=np.float32)
    nearest = np.full(N_FRAMES, -1, dtype=np.int16)
    baseline_codes = np.full(N_FRAMES, -1, dtype=np.int8)

    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    for start in range(0, len(seen), 20000):
        idx = seen[start:start + 20000]
        d = np.linalg.norm(players[idx] - ball[idx, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        k = np.argmin(d, axis=1)
        dmin = d[np.arange(len(idx)), k]
        has = np.isfinite(dmin)
        dist[idx[has]] = dmin[has]
        nearest[idx[has]] = k[has]
        baseline_codes[idx[has]] = slot_team[k[has]]
        close = has & (dmin <= params["radius_m"])
        labels[idx[close]] = np.array(list(TEAMS))[slot_team[k[close]]]
        reason[idx[close]] = 1
        reason[idx[~close]] = 2
    return labels, reason, dist, nearest, baseline_codes


# ==================================================================================================
# Step 2 -- dead ball
# ==================================================================================================

def player_pace(players):
    """Median speed of the tracked players per frame (m/s), smoothed over one second."""
    k = 5
    v = np.full(players.shape[:2], np.nan, dtype=np.float32)
    v[k:-k] = np.linalg.norm(players[2 * k:] - players[:-2 * k], axis=2) * FPS / (2 * k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.minimum(v, 12), axis=1)
    valid = ~np.isnan(med)
    kernel = np.ones(FPS)
    num = np.convolve(np.where(valid, med, 0), kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den).astype(np.float32)


def step2_dead_ball(labels1, ball, pace, restarts, params):
    """Mark stoppages D. Four rules, each with its own reason code:

      1 kick-off: everything in a half before its 'Kick Off (Start)' set piece;
      2 set piece: from the last moment the ball was seen moving fast inside the pitch (at least one
        second before), up to the restart the Smart Tagging marks (refined to the first fast ball
        inside the tag window);
      3 ball out, no tag: the ball is seen off the pitch for a few frames in a row and no tagged
        restart covers it -> D until the ball is next seen moving fast inside the pitch;
      4 play stopped: the players walk (median speed below still_speed) for at least still_min
        seconds and the ball is never seen moving fast meanwhile -- fouls, missed tags.
    """
    m = params["out_margin_m"]
    seen = ~np.isnan(ball[:, 0])
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        inside = seen & (x >= -m) & (x <= L + m) & (y >= -m) & (y <= W + m)
        v = speed_of(ball)
        # Fast, but physically possible (a jump of 20 m in a few frames is a wrong detection, not a
        # pass), and for a few frames in a row -- one glitch must not end a stoppage.
        fast = inside & (v > params["fast_ball_ms"]) & (v < 40.0)
    fast_in_play = np.zeros(N_FRAMES, dtype=bool)
    for a, b, on in runs_of(fast):
        if on and b - a >= 3:
            fast_in_play[a:b] = True
    outside = seen & ~inside
    max_dead = int(params["max_dead_s"] * FPS)

    dead = []          # dicts: start, stop (= restart frame, exclusive), why, type, team
    for r in restarts:
        half, first, last = half_of(r["hi"])
        if half is None:
            half, first, last = half_of(r["lo"])
        if half is None:
            continue
        lo, hi = max(r["lo"], first), min(r["hi"], last)
        fast = np.flatnonzero(fast_in_play[lo:hi + 1])
        restart = lo + int(fast[0]) if len(fast) else (lo + hi) // 2
        r["frame"] = restart
        if r["type"].lower() == "kick off (start)":
            dead.append({"start": first, "stop": restart, "why": 1, "type": r["type"], "team": r["team"]})
            continue
        floor = max(first, restart - max_dead)
        before = np.flatnonzero(fast_in_play[floor:max(floor, restart - FPS)])   # a set piece stops play >= 1 s
        start = floor + int(before[-1]) + 1 if len(before) else floor
        if start < restart:
            dead.append({"start": start, "stop": restart, "why": 2, "type": r["type"], "team": r["team"]})

    covered = np.zeros(N_FRAMES, dtype=bool)
    for d in dead:
        covered[d["start"]:d["stop"]] = True
    for a, b, v in runs_of(outside & ~covered):
        if not v or b - a < params["out_min_frames"]:
            continue
        half, first, last = half_of(a)
        if half is None:
            continue
        after = np.flatnonzero(fast_in_play[a + 2 * FPS:min(last + 1, a + max_dead)])
        stop = a + 2 * FPS + int(after[0]) if len(after) else min(last + 1, a + max_dead)
        nxt = [d["start"] for d in dead if d["start"] > a]
        stop = min([stop] + nxt)
        if stop > a:
            dead.append({"start": a, "stop": stop, "why": 3, "type": "fuera sin tag", "team": "?"})
            covered[a:stop] = True

    with np.errstate(invalid="ignore"):
        still = (pace < params["still_speed_ms"]) & ~covered & graded_mask()
    fast_count = np.concatenate(([0], np.cumsum(fast_in_play)))
    for a, b, v in runs_of(still):
        if v and b - a >= params["still_min_s"] * FPS and fast_count[b] == fast_count[a]:
            dead.append({"start": a, "stop": b, "why": 4, "type": "juego parado", "team": "?"})

    labels = labels1.copy()
    reason = np.zeros(N_FRAMES, dtype=np.int8)
    for d in sorted(dead, key=lambda d: d["why"], reverse=True):     # kick-off wins overlaps
        labels[d["start"]:d["stop"]] = "D"
        reason[d["start"]:d["stop"]] = d["why"]
    dead.sort(key=lambda d: d["start"])
    return labels, reason, dead


# ==================================================================================================
# Step 3 -- control, not touches
# ==================================================================================================

def step3_control(labels2, params):
    """Inside each live stretch, a team run shorter than min_control is a touch: give it back.

    A run is consecutive observations of one team; '-' frames do not break it (a pass in the air
    between two team-mates is still one run). Shortest runs are absorbed first, until every
    remaining run lasts at least min_control or the stretch has a single run.
    """
    min_frames = int(round(params["min_control_s"] * FPS))
    labels = labels2.copy()
    reason = np.zeros(N_FRAMES, dtype=np.int8)
    removed, kept = [], []
    for a, b in live_stretches(labels2):
        obs = a + np.flatnonzero(np.isin(labels2[a:b], ("H", "A")))
        if not len(obs):
            continue
        team = labels2[obs]
        cut = np.flatnonzero(team[1:] != team[:-1]) + 1
        bounds = np.concatenate(([0], cut, [len(obs)]))
        runs = [[team[i], [(int(obs[i]), int(obs[j - 1]))]] for i, j in zip(bounds[:-1], bounds[1:])]
        span = lambda run: run[1][-1][1] - run[1][0][0] + 1   # noqa: E731
        while len(runs) > 1:
            k = min(range(len(runs)), key=lambda i: span(runs[i]))
            if span(runs[k]) >= min_frames:
                break
            new = runs[k + 1][0] if k == 0 else runs[k - 1][0]
            for p, q in runs[k][1]:
                removed.append((p, q + 1, runs[k][0], new, (q - p + 1) / FPS))
            runs[k][0] = new
            merged = []
            for run in runs:
                if merged and merged[-1][0] == run[0]:
                    merged[-1][1].extend(run[1])
                else:
                    merged.append([run[0], list(run[1])])
            runs = merged
        for run in runs:
            kept.append((run[1][0][0], run[1][-1][1] + 1, run[0]))
    for p, q, old, new, _ in removed:          # applied in order: later merges override earlier ones
        seg = labels[p:q]
        seg[seg == old] = new
    changed = (labels != labels2) & np.isin(labels2, ("H", "A"))
    reason[changed] = 1
    removed = [r for r in removed if changed[r[0]:r[1]].any() and labels[r[0]] == r[3]]
    return labels, reason, removed, kept


# ==================================================================================================
# Step 4 -- gaps without a ball
# ==================================================================================================

def advance_signal(players, slot_team):
    """Per frame, how fast each team moves towards the goal it attacks (m/s), averaged over slots.

    The attacking direction of each team in each half is read from the data itself: a team's
    players sit on average on the side of the goal they defend.
    """
    vx = np.full(players.shape[:2], np.nan, dtype=np.float32)
    k = 5
    vx[k:-k] = (players[2 * k:, :, 0] - players[:-2 * k, :, 0]) * FPS / (2 * k)
    vx = np.clip(vx, -9, 9)
    home = slot_team == 0
    direction = {}
    adv = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for half, first, last in HALVES:
            sl = slice(first, last + 1)
            hx = np.nanmean(players[sl][:, home, 0], axis=1)
            ax = np.nanmean(players[sl][:, ~home, 0], axis=1)
            sign = 1.0 if np.nanmean(ax - hx) > 0 else -1.0       # +1: home attacks towards x = L
            direction[half] = sign
            adv[sl, 0] = sign * np.nanmean(vx[sl][:, home], axis=1)
            adv[sl, 1] = -sign * np.nanmean(vx[sl][:, ~home], axis=1)
    return adv, direction


def step4_gaps(labels3, adv, params):
    """Fill every '-' inside a live stretch.

      1 same team before and after the gap  -> that team;
      2/3 different teams                   -> split the gap where the new team starts advancing
                                               (frames before the cut -> old team, after -> new);
      4 start or end of a stretch           -> the nearest observation;
      5 no observation in the whole stretch -> X, the scorer carries the previous label forward.
    """
    labels = labels3.copy()
    reason = np.zeros(N_FRAMES, dtype=np.int8)
    gaps = []
    min_split = int(params["min_split_gap_s"] * FPS)
    for a, b in live_stretches(labels3):
        obs = a + np.flatnonzero(np.isin(labels3[a:b], ("H", "A")))
        if not len(obs):
            labels[a:b], reason[a:b] = "X", 5
            gaps.append((a, b, "?", "?", -1, 5))
            continue
        if obs[0] > a:
            labels[a:obs[0]], reason[a:obs[0]] = labels3[obs[0]], 4
            gaps.append((a, int(obs[0]), "?", labels3[obs[0]], -1, 4))
        if obs[-1] + 1 < b:
            labels[obs[-1] + 1:b], reason[obs[-1] + 1:b] = labels3[obs[-1]], 4
            gaps.append((int(obs[-1]) + 1, b, labels3[obs[-1]], "?", -1, 4))
        holes = np.flatnonzero(np.diff(obs) > 1)
        for i in holes:
            g0, g1 = int(obs[i]) + 1, int(obs[i + 1])           # gap is [g0, g1)
            before, after = labels3[obs[i]], labels3[obs[i + 1]]
            if before == after:
                labels[g0:g1], reason[g0:g1] = before, 1
                gaps.append((g0, g1, before, after, -1, 1))
                continue
            split = (g0 + g1) // 2
            if g1 - g0 >= min_split:
                new, old = TEAMS.index(after), TEAMS.index(before)
                e = adv[g0:g1, new] - adv[g0:g1, old]
                if np.isfinite(e).sum() > (g1 - g0) // 2:
                    prefix = np.concatenate(([0.0], np.cumsum(np.nan_to_num(e))))
                    split = g0 + int(np.argmin(prefix))
            labels[g0:split], reason[g0:split] = before, 2
            labels[split:g1], reason[split:g1] = after, 3
            gaps.append((g0, g1, before, after, split, 2))
    return labels, reason, gaps


# ==================================================================================================
# Scoring and diagnostics
# ==================================================================================================

def score(truth, labels, kind):
    gt = pm.label_codes(truth)
    r = pm.evaluate(gt, pm.label_codes(labels))
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_poss": r["n_pred_spells"],
            "dur": r["mean_spell_duration_s"], "d_f1": r["per_class"]["D"]["f1"],
            "n_gt": r["n_gt_spells"], "gt_dur": r["gt_mean_spell_duration_s"]}


def diagnostics(labels):
    """What a step's output looks like without any truth: counts and shares only."""
    graded = graded_mask()
    filled = labels.copy()
    out = {"per_half": []}
    poss, lengths = 0, []
    for half, first, last in HALVES:
        seg = labels[first:last + 1]
        known = np.flatnonzero(seg != "X")
        if len(known):
            carry = np.maximum.accumulate(np.where(seg != "X", np.arange(len(seg)), -1))
            carry[carry < 0] = known[0]
            seg = seg[carry]
        filled[first:last + 1] = seg
        n = [b - a for a, b, v in runs_of(seg) if v in ("H", "A")]
        out["per_half"].append(len(n))
        poss += len(n)
        lengths += n
    g = labels[graded]
    out.update({"n_poss": poss, "dur": float(np.mean(lengths)) / FPS if lengths else 0.0,
                "dead_pct": 100.0 * np.mean(g == "D"), "none_pct": 100.0 * np.mean(g == "X"),
                "home_pct": 100.0 * np.sum(filled[graded] == "H") / max(1, np.sum(np.isin(filled[graded], ("H", "A"))))})
    return out


def error_analysis(truth, final):
    """Where the final step disagrees with the truth, by kind, plus how late team changes are."""
    graded = truth != "X"
    kind = np.full(N_FRAMES, "", dtype=object)
    team_t, team_p = np.isin(truth, ("H", "A")), np.isin(final, ("H", "A"))
    kind[graded & (truth == "D") & team_p] = "D no detectado"
    kind[graded & team_t & (final == "D")] = "D inventado"
    kind[graded & (final == "X")] = "Sin respuesta"
    wrong_team = graded & team_t & team_p & (truth != final)
    kind[wrong_team] = "Equipo equivocado"

    segments = []
    for a, b, v in runs_of(kind):
        if not v:
            continue
        label = v
        if v == "Equipo equivocado":
            edge = (a > 0 and truth[a - 1] != truth[a]) or (b < N_FRAMES and truth[b] != truth[b - 1])
            label = "Cambio desplazado" if edge else "Posesión inventada o perdida"
        segments.append({"start": a, "stop": b, "type": label, "truth": str(truth[a]), "pred": str(final[a])})
    by_type = {}
    for s in segments:
        t = by_type.setdefault(s["type"], {"type": s["type"], "frames": 0, "count": 0})
        t["frames"] += s["stop"] - s["start"]
        t["count"] += 1
    total = int(graded.sum())
    by_type = sorted(by_type.values(), key=lambda t: -t["frames"])
    for t in by_type:
        t["pct"] = 100.0 * t["frames"] / total
    top = sorted(segments, key=lambda s: s["start"] - s["stop"])[:15]
    for s in top:
        s["when"] = clock(s["start"])
        s["seconds"] = (s["stop"] - s["start"]) / FPS
        half, first, _ = half_of(s["start"])
        s["cmd"] = "--window-half %d --window-minute %.1f" % (half, max(0.0, (s["start"] - first) / FPS / 60.0 - 0.5))

    gt_codes = pm.label_codes(truth)
    runs = pm.valid_runs(gt_codes)
    pred_codes = pm.fill_abstentions(pm.label_codes(final), runs)
    gt_tr = [t for t in pm.transitions(gt_codes, runs) if t[1] in (0, 1) and t[2] in (0, 1)]
    pr_tr = [t for t in pm.transitions(pred_codes, runs) if t[1] in (0, 1) and t[2] in (0, 1)]
    pr_frames = {}
    for f, a, b in pr_tr:
        pr_frames.setdefault((a, b), []).append(f)
    offsets, missed = [], 0
    for f, a, b in gt_tr:
        cands = [p for p in pr_frames.get((a, b), []) if abs(p - f) <= 5 * FPS]
        if not cands:
            missed += 1
            continue
        offsets.append((min(cands, key=lambda p: abs(p - f)) - f) / FPS)
    edges = np.arange(-5.0, 5.01, 0.5)
    counts, _ = np.histogram(np.clip(offsets, -4.999, 4.999), bins=edges)
    return {"by_type": by_type, "top": top, "graded": total,
            "offsets": {"edges": edges.tolist(), "counts": counts.tolist(), "missed": missed,
                        "n": len(gt_tr), "median": float(np.median(offsets)) if offsets else None,
                        "within_half_s": int(np.sum(np.abs(offsets) <= 0.5))}}


# ==================================================================================================
# Report
# ==================================================================================================

def per_second(labels, truth_like=False):
    """One character per second of each graded half (the label held by most frames)."""
    out = []
    for _, first, last in HALVES:
        seg = labels[first:last + 1]
        n = len(seg) // FPS
        block = seg[:n * FPS].reshape(n, FPS)
        chars = []
        for row in block:
            values, counts = np.unique(row, return_counts=True)
            chars.append(values[np.argmax(counts)])
        out.append("".join(chars))
    return out


def step_texts(p, s1, s2, s3, s4):
    r = p["radius_m"]
    return [
        {"q": "¿Quién está más cerca del balón?",
         "data": ["ATD: posición del balón", "ATD: posición de los 46 slots", "XML: equipo de cada slot"],
         "rule": "Si el jugador más cercano al balón está a ≤ %.1f m, su equipo tiene el balón. Si no se ve el balón, "
                 "o se ve pero nadie está cerca (suele ser un pase en el aire), el paso responde «—»: no lo sabe." % r,
         "stats": s1},
        {"q": "¿Está el balón en juego?",
         "data": ["ATD: balón fuera de las líneas", "ATD: velocidad del balón", "ATD: velocidad de los jugadores",
                  "Smart Tagging: SET PIECES"],
         "rule": "Cada balón parado del Smart Tagging marca una reanudación. Desde la última vez que vimos el balón "
                 "moverse rápido (> %.0f m/s) dentro del campo hasta esa reanudación, el balón está muerto: D. Antes del "
                 "saque inicial de cada parte, también D. Si el balón se ve fuera del campo y no hay tag, D hasta que "
                 "vuelva a moverse rápido dentro. Y si los jugadores caminan (velocidad mediana < %.1f m/s) durante "
                 "%.0f s o más sin que el balón se mueva rápido, el juego está parado: D (faltas y tags que faltan)." % (
                     p["fast_ball_ms"], p["still_speed_ms"], p["still_min_s"]),
         "stats": s2},
        {"q": "¿Es control o solo un toque?",
         "data": ["La salida del paso 2 a lo largo del tiempo"],
         "rule": "Dentro de cada tramo en juego, una racha de un equipo que dura menos de %.1f s no es control: es un "
                 "toque, un desvío o ruido del tracking, y se devuelve al equipo que tenía el balón. Los «—» no cortan "
                 "una racha: un pase en el aire entre compañeros sigue siendo la misma racha." % p["min_control_s"],
         "stats": s3},
        {"q": "¿Qué pasa cuando no se ve el balón?",
         "data": ["La salida del paso 3", "ATD: velocidad de los jugadores hacia la portería rival"],
         "rule": "Cada hueco «—» dentro de un tramo en juego se rellena. Si antes y después tiene el balón el mismo "
                 "equipo, se rellena con él. Si cambia, el corte se pone donde el equipo nuevo empieza a avanzar "
                 "(y el antiguo a replegarse). Si un tramo no tiene ninguna observación, se escribe X.",
         "stats": s4},
    ]


def es_numbers(step):
    """Spanish decimal commas in the report text: 2.0 m -> 2,0 m."""
    fix = lambda text: re.sub(r"(\d)\.(\d)", r"\1,\2", text)   # noqa: E731
    return dict(step, rule=fix(step["rule"]), stats=[fix(t) for t in step["stats"]])


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def build_report(ctx):
    """Assemble the JSON the HTML template reads, and write report.html."""
    p, graded = ctx["params"], graded_mask()
    g = int(graded.sum())
    l1, r1, l2, r2, l3, r3, l4, r4 = (ctx[k] for k in ("l1", "r1", "l2", "r2", "l3", "r3", "l4", "r4"))

    seen = int(np.sum(graded & (r1 > 0)))
    close = int(np.sum(graded & (r1 == 1)))
    s1 = ["El balón se ve en el %.1f%% de los frames evaluados." % pct(seen, g),
          "Hay un jugador a ≤ %.1f m en el %.1f%%: ahí el paso dice H (%.1f%%) o A (%.1f%%)." % (
              p["radius_m"], pct(close, g), pct(np.sum(graded & (l1 == "H")), g), pct(np.sum(graded & (l1 == "A")), g)),
          "Se ve el balón pero nadie está cerca en el %.1f%%." % pct(np.sum(graded & (r1 == 2)), g),
          "En total responde «—» (no lo sé) en el %.1f%% de los frames." % pct(np.sum(graded & (l1 == NONE)), g)]
    types = {}
    for r in ctx["restarts"]:
        if "frame" in r:
            types[r["type"]] = types.get(r["type"], 0) + 1
    dead = ctx["dead"]
    newly_d = graded & (l2 == "D")
    s2 = ["Reanudaciones encontradas en el Smart Tagging: %d (%s)." % (
              sum(types.values()), ", ".join("%s %d" % (k, v) for k, v in sorted(types.items(), key=lambda kv: -kv[1]))),
          "Tramos de balón muerto: %d antes de una reanudación, %d antes de un saque inicial, %d por balón fuera sin tag "
          "y %d de juego parado." % (sum(d["why"] == 2 for d in dead), sum(d["why"] == 1 for d in dead),
                                     sum(d["why"] == 3 for d in dead), sum(d["why"] == 4 for d in dead)),
          "Frames marcados D: %.1f%% de los evaluados; en el paso 1, el %.1f%% de ellos tenían H o A." % (
              pct(newly_d.sum(), g), pct(np.sum(newly_d & np.isin(l1, ("H", "A"))), newly_d.sum()))]
    s3 = ["Rachas cortas (< %.1f s) descartadas: %d, con %d frames observados que cambian de equipo." % (
              p["min_control_s"], len(ctx["removed"]), int(np.sum(r3 > 0))),
          "Posesiones antes y después de este paso: %d → %d." % (ctx["diag"][1]["n_poss"], ctx["diag"][2]["n_poss"])]
    gaps = ctx["gaps"]
    same = [x for x in gaps if x[5] == 1]
    change = [x for x in gaps if x[5] == 2]
    edge = [x for x in gaps if x[5] == 4]
    empty = [x for x in gaps if x[5] == 5]
    s4 = ["Huecos con el mismo equipo antes y después: %d (%.1f%% de los frames), rellenados." % (
              len(same), pct(sum(x[1] - x[0] for x in same), g)),
          "Huecos con cambio de equipo: %d (%.1f%% de los frames), cortados donde el equipo nuevo empieza a avanzar." % (
              len(change), pct(sum(x[1] - x[0] for x in change), g)),
          "Inicios y finales de tramo sin observación: %d. Tramos sin ninguna observación (X): %d." % (len(edge), len(empty))]

    # explorer window
    w0, w1 = ctx["window"]
    step = 2
    sample = np.arange(w0, w1, step)
    pos = np.round(ctx["players"][sample] * 10)
    pos = [[None if np.isnan(v) else int(v) for v in row.ravel()] for row in pos]
    bpos = [None if np.isnan(b[0]) else [int(round(b[0] * 10)), int(round(b[1] * 10))] for b in ctx["ball"][sample]]
    truth = ctx["truth"]

    def win(arr):
        return "".join(arr[w0:w1].tolist())

    def win_digits(arr):
        return "".join(str(int(v)) for v in arr[w0:w1])

    dist = ctx["dist"][w0:w1]
    rows = {"s1": win(l1), "s2": win(l2), "s3": win(l3), "s4": win(for_scoring(l4)),
            "ball": "".join("V" if v else "-" for v in ~np.isnan(ctx["ball"][w0:w1, 0]))}
    if truth is not None:
        rows["truth"] = win(truth)
    inwin = lambda a, b: b > w0 and a < w1   # noqa: E731
    window = {
        "start": w0, "end": w1, "step": step, "slot_team": ctx["slot_team"].tolist(), "pos": pos, "ball": bpos,
        "nearest": ctx["nearest"][sample].tolist(), "dist": [None if np.isnan(v) else round(float(v), 1) for v in dist],
        "pace": [None if np.isnan(v) else round(float(v), 1) for v in ctx["pace"][w0:w1]],
        "rows": rows,
        "reasons": {"s1": win_digits(r1), "s2": win_digits(r2), "s3": win_digits(r3), "s4": win_digits(r4)},
        "dead": [[d["start"], d["stop"], d["why"], d["type"], d["team"]] for d in dead if inwin(d["start"], d["stop"])],
        "removed": [[a, b, o, n, round(s, 2)] for a, b, o, n, s in ctx["removed"] if inwin(a, b)],
        "kept": [[a, b, t] for a, b, t in ctx["kept"] if inwin(a, b)],
        "gaps": [[a, b, bf, af, sp, k] for a, b, bf, af, sp, k in gaps if inwin(a, b)],
        "last_fast": {},
    }
    def example(arr, preferred):
        """Middle of the longest run of the first preferred reason found in the window."""
        for codes in preferred:
            runs = [(b - a, a, b) for a, b, v in runs_of(np.isin(arr[w0:w1], codes)) if v]
            if runs:
                _, a, b = max(runs)
                return int(w0 + (a + b) // 2)
        return None
    examples = [example(r1, [(1,)]), example(r2, [(2,), (4,), (3,), (1,)]),
                example(r3, [(1,)]), example(r4, [(2, 3), (1,), (4,)])]

    overview = {"rows": [{"name": "Balón visible", "s": per_second(np.where(np.isnan(ctx["ball"][:, 0]), "-", "V"))}]}
    for name, arr in zip(STEP_NAMES, (l1, l2, l3, for_scoring(l4))):
        overview["rows"].append({"name": name, "s": per_second(arr)})
    if truth is not None:
        overview["rows"].append({"name": "Verdad" if ctx["truth_kind"] == "full" else "Verdad (5 min públicos)",
                                 "s": per_second(truth)})

    data = {
        "meta": {"source": ctx["source"], "is_real": ctx["is_real"], "truth_kind": ctx["truth_kind"],
                 "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "params": p,
                 "fps": FPS, "halves": [list(h) for h in HALVES], "pitch": [L, W],
                 "direction": {str(k): v for k, v in ctx["direction"].items()}, "window_clock": [clock(w0), clock(w1 - 1)]},
        "metrics": ctx["metrics"], "diag": ctx["diag_rows"],
        "steps": [dict(es_numbers(t), name=n, example=e)
                  for t, n, e in zip(step_texts(p, s1, s2, s3, s4), STEP_NAMES, examples)],
        "overview": overview, "window": window, "errors": ctx["errors"],
    }
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    path = os.path.join(ctx["out_dir"], "report.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path


# ==================================================================================================
# Main
# ==================================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or games.DATA_DIR,
                        help="the data bundle (default: $MTAH_DATA_DIR, else ./data)")
    parser.add_argument("--out", default=None, help="output folder (default: <data-dir>/reports/heuristics)")
    parser.add_argument("--window-half", type=int, default=1, choices=(1, 2), help="explorer: which half")
    parser.add_argument("--window-minute", type=float, default=0.0, help="explorer: minute of that half to start")
    parser.add_argument("--window-seconds", type=int, default=180, help="explorer: length (max 600)")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    params = {k: getattr(args, k) for k in DEFAULTS}

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out or os.path.join(data_dir, "reports", "heuristics"))
    paths = dict(zip(["atd_meta", "atd_raw"], games.require(["atd_meta", "atd_raw"], GAME, data_dir)))
    truth, truth_kind = load_truth(data_dir)
    is_real = truth_kind != "full"
    t0 = time.time()

    players, ball, slot_team = load_tracking(paths, os.path.join(out_dir, "cache", "atd.npz"))
    restarts = load_restarts(data_dir, params)
    print("loaded tracking and tagging           %5.1f s" % (time.time() - t0))

    l1, r1, dist, nearest, base_codes = step1_nearest(players, ball, slot_team, params)
    pace = player_pace(players)
    l2, r2, dead = step2_dead_ball(l1, ball, pace, restarts, params)
    l3, r3, removed, kept = step3_control(l2, params)
    adv, direction = advance_signal(players, slot_team)
    l4, r4, gaps = step4_gaps(l3, adv, params)
    print("ran the four steps                    %5.1f s" % (time.time() - t0))

    baseline = baseline_smooth(base_codes, 50)
    outputs = [("Baseline oficial (smooth50)", "baseline_smooth50", baseline)]
    outputs += [(n, f, for_scoring(lab)) for n, f, lab in zip(STEP_NAMES, STEP_FILES, (l1, l2, l3, l4))]
    for _, fname, lab in outputs:
        write_csv(os.path.join(out_dir, "submissions", fname + ".csv"), lab)

    metrics = None
    if truth is not None:
        metrics = [dict(score(truth, lab, truth_kind), name=n) for n, _, lab in outputs]
    diag = [diagnostics(lab) for _, _, lab in outputs[1:]]
    diag_rows = [dict(d, name=n) for d, (n, _, _) in zip([diagnostics(baseline)] + diag, outputs)]
    errors = error_analysis(truth, for_scoring(l4)) if truth_kind == "full" else None

    half, first, last = HALVES[args.window_half - 1]
    w0 = min(last, first + int(args.window_minute * 60 * FPS))
    w1 = min(last + 1, w0 + min(600, args.window_seconds) * FPS)
    ctx = dict(params=params, l1=l1, r1=r1, l2=l2, r2=r2, l3=l3, r3=r3, l4=l4, r4=r4, dist=dist, nearest=nearest,
               dead=dead, pace=pace, removed=removed, kept=kept, gaps=gaps, restarts=restarts, players=players, ball=ball,
               slot_team=slot_team, truth=truth, truth_kind=truth_kind, direction=direction, window=(w0, w1),
               metrics=metrics, diag=diag, diag_rows=diag_rows, errors=errors, out_dir=out_dir, is_real=is_real,
               source=os.path.relpath(data_dir, ROOT) if not is_real else "datos reales")
    report = build_report(ctx)

    summary = {"params": params, "diagnostics": diag_rows, "metrics": metrics, "truth": truth_kind}
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print("\n%-30s %8s %8s %6s %7s" % ("", "spellF1", "macroF1", "poss", "dur s"))
    for i, (name, _, _) in enumerate(outputs):
        m = metrics[i] if metrics else None
        d = diag_rows[i]
        print("%-30s %8s %8s %6d %7.1f" % (name, "%.4f" % m["spell"] if m else "-", "%.4f" % m["macro"] if m else "-",
                                          d["n_poss"], d["dur"]))
    if truth_kind == "public":
        print("\n(truth = the public 5 minutes: a pipeline check only, never a reason to change a parameter)")
    print("\nreport:      %s" % report)
    print("submissions: %s" % os.path.join(out_dir, "submissions"))
    print("total %.1f s" % (time.time() - t0))


# ==================================================================================================
# The report template. Self-contained: no external scripts, fonts or images.
# ==================================================================================================

TEMPLATE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Posesión por heurísticas</title>
<style>
:root{--bg:#faf9f6;--card:#ffffff;--ink:#1f1e1c;--mut:#6b6a66;--faint:#a3a19b;--line:#e4e2dc;--soft:#f1efe8;
--H:#378ADD;--A:#D85A30;--D:#888780;--N:#d9d7cf;--X:#efede6;--ok:#3B6D11;--bad:#A32D2D;--warnbg:#FAEEDA;--warn:#633806;--pitch:#eef3ea;--pline:#b9c7b0}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--card:#242422;--ink:#ecebe6;--mut:#a7a59e;--faint:#77756f;--line:#3a3936;--soft:#2e2d2a;
--N:#4a4945;--X:#33322f;--ok:#97C459;--bad:#F09595;--warnbg:#412402;--warn:#FAC775;--pitch:#1f2a1e;--pline:#47573f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 18px 80px}h1{font-size:26px;font-weight:600;margin:0 0 4px}
h2{font-size:19px;font-weight:600;margin:44px 0 6px}h3{font-size:15px;font-weight:600;margin:0}p{margin:6px 0}
.sub{color:var(--mut);font-size:13.5px}.lead{color:var(--mut);max-width:780px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.banner{background:var(--warnbg);color:var(--warn);border-radius:10px;padding:10px 14px;margin:14px 0;font-size:14px}
.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:14px}
.flow .card{position:relative}.flow .n{font-size:12px;color:var(--mut)}.flow .q{font-size:14px;margin:4px 0 8px}
.chip{display:inline-block;font-size:11.5px;background:var(--soft);color:var(--mut);border-radius:6px;padding:1px 7px;margin:2px 4px 0 0}
.pill{display:inline-block;min-width:22px;text-align:center;border-radius:5px;color:#fff;font-weight:600;font-size:12px;padding:1px 6px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--mut);font-size:12.5px}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{height:8px;border-radius:4px;background:var(--H);display:inline-block;vertical-align:middle;margin-right:6px}
.best td{background:var(--soft)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--mut);margin:8px 0}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-2px}
.ex{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(0,1fr);gap:14px;align-items:start}
canvas{display:block;width:100%}button{font:inherit;font-size:13px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:5px 11px;cursor:pointer}
button:hover{border-color:var(--faint)}button.on{background:var(--soft)}
.ctrl{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0}.ctrl input[type=range]{flex:1;min-width:160px}
.clock{font-variant-numeric:tabular-nums;font-size:14px;font-weight:600;min-width:170px}
.why{display:grid;grid-template-columns:150px minmax(0,1fr);gap:4px 10px;font-size:13.5px}
.why .k{color:var(--mut);padding-top:2px}.why .v{padding-bottom:8px;border-bottom:1px solid var(--line)}
.steps{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.steps ul{margin:6px 0 0;padding-left:18px;color:var(--mut);font-size:13.5px}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:4px 8px;border-radius:6px;display:none;z-index:9;white-space:nowrap}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
@media (max-width:860px){.flow,.steps,.grid2{grid-template-columns:1fr}.ex{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Posesión por heurísticas</h1>
<div class="sub" id="subtitle"></div>
<div id="banners"></div>

<h2>Cómo funciona</h2>
<p class="lead">Cuatro pasos encadenados. Cada uno responde una sola pregunta con unos datos concretos y corrige un tipo de error. Lo que un paso no sabe decidir lo deja como «—» (sin respuesta) para el siguiente.</p>
<div class="flow" id="flow"></div>

<h2>Qué consigue cada paso</h2>
<p class="lead" id="metrics-lead"></p>
<div class="card" style="padding:6px 10px"><table id="metrics"></table></div>

<h2>El partido entero</h2>
<p class="lead">Cada columna de píxeles es un trozo del partido; cada fila, la salida de un paso. El recuadro marca la ventana del explorador. Pasa el ratón por encima para ver el minuto y haz clic dentro del recuadro para saltar a ese momento.</p>
<div class="legend" id="legend"></div>
<div class="card"><canvas id="overview"></canvas></div>

<h2>Explorador frame a frame</h2>
<p class="lead">Una ventana de <span id="wlen"></span> (<span id="wclock"></span>). A la izquierda, el campo tal como lo ve el tracking; a la derecha, qué ha decidido cada paso en este frame y por qué. Debajo, la línea de tiempo: haz clic para saltar, usa ←/→ para avanzar frame a frame y la barra espaciadora para reproducir.</p>
<div class="card">
<div class="ctrl"><button id="play">▶ Reproducir</button><button id="sp">1×</button><button id="zoom">Zoom 30 s</button>
<input type="range" id="slider"><span class="clock" id="clock"></span></div>
<div class="ex"><div><canvas id="pitch"></canvas><div class="legend" style="margin-top:6px"><span><i style="background:var(--H)"></i>Barça (H)</span><span><i style="background:var(--A)"></i>Athletic (A)</span><span><i style="background:var(--ink);border-radius:50%"></i>balón</span><span>anillo: jugador más cercano</span></div></div>
<div class="why" id="why"></div></div>
<canvas id="timeline" style="margin-top:14px"></canvas>
</div>

<h2>Paso a paso</h2>
<div class="steps" id="steps"></div>

<div id="errsec"></div>

<h2>Por dónde seguir</h2>
<div class="card" id="next"></div>
</main><div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const FPS=D.meta.fps, W0=D.window.start, W1=D.window.end, NW=W1-W0, P=D.meta.params;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL=()=>({H:css('--H'),A:css('--A'),D:css('--D'),'-':css('--N'),X:css('--X'),V:css('--ok')});
const NAMES={H:'H · Barça',A:'A · Athletic',D:'D · balón muerto','-':'— · sin respuesta',X:'X · no respondo'};
const fmt=(v,d=1)=>v==null?'—':Number(v).toLocaleString('es-ES',{minimumFractionDigits:d,maximumFractionDigits:d});
function clock(f){for(const [h,a,b] of D.meta.halves){if(f>=a&&f<=b){const t=Math.round((f-a)*10/FPS);return h+'ª parte '+String(Math.floor(t/600)).padStart(2,'0')+':'+String(Math.floor(t/10)%60).padStart(2,'0')+'.'+(t%10)}}return '—'}
function pill(l){const c=COL();return '<span class="pill" style="background:'+(c[l]||c.X)+';color:'+(l==='-'||l==='X'?css('--ink'):'#fff')+'">'+(l==='-'?'—':l)+'</span>'}
const tip=document.getElementById('tip');
function showTip(e,t){tip.style.display='block';tip.textContent=t;tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+14)+'px'}
function hideTip(){tip.style.display='none'}
function sizeCanvas(cv,h){const r=window.devicePixelRatio||1,w=cv.clientWidth;cv.width=w*r;cv.height=h*r;cv.style.height=h+'px';const x=cv.getContext('2d');x.setTransform(r,0,0,r,0,0);return [x,w]}

document.getElementById('subtitle').textContent='Datos: '+D.meta.source+' · generado '+D.meta.generated+' · radio '+fmt(P.radius_m)+' m · control mínimo '+fmt(P.min_control_s)+' s · balón rápido > '+fmt(P.fast_ball_ms,0)+' m/s';
let B='';
if(D.meta.is_real)B+='<div class="banner">Este informe se ha generado con los datos reales y contiene extractos del tracking. Guárdalo en tu ordenador: no lo subas a ningún repositorio ni se lo pases a un LLM en la nube.</div>';
if(D.meta.truth_kind==='public')B+='<div class="banner">La «verdad» solo cubre los 5 minutos públicos. Sirve para comprobar que el pipeline es correcto, no para elegir parámetros (RULES.md).</div>';
if(D.meta.truth_kind==='full')B+='<div class="banner">Datos sintéticos con verdad completa. Los números sirven para entender qué aporta cada paso y encontrar fallos, no para elegir parámetros: el simulador tiene sus propios sesgos.</div>';
document.getElementById('banners').innerHTML=B;

document.getElementById('flow').innerHTML=D.steps.map((s,i)=>'<div class="card"><div class="n">Paso '+(i+1)+'</div><h3>'+s.name.split('· ')[1].replace(/^./,c=>c.toUpperCase())+'</h3><div class="q">'+s.q+'</div>'+s.data.map(d=>'<span class="chip">'+d+'</span>').join('')+'</div>').join('');

(function metrics(){
 const t=document.getElementById('metrics'),M=D.metrics,G=D.diag;
 if(M){
  const best=Math.max(...M.slice(1).map(m=>m.spell)),base=M[0].spell,last=M[M.length-1].spell;
  document.getElementById('metrics-lead').innerHTML='Cada fila es la salida de un paso, puntuada con el scorer oficial contra '+(D.meta.truth_kind==='full'?'la verdad completa de los datos sintéticos':'los 5 minutos públicos')+'. El ranking usa <b>spellF1</b> (posesiones enteras); macroF1 es el desempate. El paso 4 '+(last>base?'supera':'no supera')+' al baseline oficial: '+fmt(last,3)+' frente a '+fmt(base,3)+'.';
  t.innerHTML='<tr><th>Salida</th><th>spellF1</th><th>macroF1</th><th>Posesiones</th><th>Duración media</th><th>F1 de D</th></tr>'+M.map((m,i)=>'<tr class="'+(i>0&&m.spell===best?'best':'')+'"><td>'+m.name+'</td><td class="num"><span class="bar" style="width:'+Math.max(2,m.spell*150)+'px"></span>'+fmt(m.spell,3)+'</td><td class="num"><span class="bar" style="background:var(--D);width:'+Math.max(2,m.macro*150)+'px"></span>'+fmt(m.macro,3)+'</td><td class="num">'+m.n_poss+'</td><td class="num">'+fmt(m.dur)+' s</td><td class="num">'+fmt(m.d_f1,2)+'</td></tr>').join('')+'<tr><td><b>Verdad</b></td><td></td><td></td><td class="num"><b>'+M[0].n_gt+'</b></td><td class="num"><b>'+fmt(M[0].gt_dur)+' s</b></td><td></td></tr>';
 } else {
  document.getElementById('metrics-lead').textContent='Sin verdad para comparar: solo diagnósticos. Compara el número y la duración de las posesiones con valores típicos de otros partidos.';
  t.innerHTML='<tr><th>Salida</th><th>Posesiones (1ª / 2ª)</th><th>Duración media</th><th>% D</th><th>% sin respuesta</th><th>% H</th></tr>'+G.map(d=>'<tr><td>'+d.name+'</td><td class="num">'+d.n_poss+' ('+d.per_half.join(' / ')+')</td><td class="num">'+fmt(d.dur)+' s</td><td class="num">'+fmt(d.dead_pct)+'%</td><td class="num">'+fmt(d.none_pct)+'%</td><td class="num">'+fmt(d.home_pct)+'%</td></tr>').join('');
 }
})();

document.getElementById('legend').innerHTML=['H','A','D','-','X'].map(l=>'<span><i style="background:var(--'+(l==='-'?'N':l)+')"></i>'+NAMES[l]+'</span>').join('')+'<span><i style="background:var(--ok)"></i>balón visible</span>';

let seek=()=>{};
(function overview(){
 const cv=document.getElementById('overview'),rows=D.overview.rows,H=[22,...rows.slice(1).map(()=>22)];
 const lab=150,gap=14,rh=20,top=18;
 let geo=null;
 function draw(){
  const [x,w]=sizeCanvas(cv,top+rows.length*(rh+6)+4),c=COL();
  const lens=rows[0].s.map(s=>s.length),tot=lens[0]+lens[1],pw=(w-lab-gap)/tot;
  geo={pw,lens,lab,gap};
  x.font='12px system-ui';x.fillStyle=css('--mut');
  x.fillText('1ª parte',lab,12);x.fillText('2ª parte',lab+lens[0]*pw+gap,12);
  rows.forEach((r,i)=>{const y=top+i*(rh+6);x.fillStyle=css('--mut');x.fillText(r.name,0,y+14);
   r.s.forEach((s,h)=>{const x0=lab+(h?lens[0]*pw+gap:0);for(let k=0;k<s.length;k++){const ch=s[k];x.fillStyle=i===0?(ch==='V'?c.V:c.X):(c[ch]||c.X);x.fillRect(x0+k*pw,y,Math.ceil(pw)+0.5,rh)}})});
  const half=D.meta.halves.findIndex(h=>W0>=h[1]&&W0<=h[2]),h0=D.meta.halves[half][1],xs=lab+(half?lens[0]*pw+gap:0);
  x.strokeStyle=css('--ink');x.lineWidth=1.5;x.strokeRect(xs+(W0-h0)/FPS*pw,top-4,NW/FPS*pw,rows.length*(rh+6));
 }
 function at(e){const r=cv.getBoundingClientRect(),px=e.clientX-r.left;if(!geo||px<geo.lab)return null;let s=(px-geo.lab)/geo.pw,h=0;if(s>geo.lens[0]){s=(px-geo.lab-geo.lens[0]*geo.pw-geo.gap)/geo.pw;h=1}if(s<0||s>geo.lens[h])return null;return D.meta.halves[h][1]+Math.floor(s*FPS)}
 cv.addEventListener('mousemove',e=>{const f=at(e);if(f==null)return hideTip();const r=cv.getBoundingClientRect(),row=Math.floor((e.clientY-r.top-top)/(rh+6));let t=clock(f);if(row>=0&&row<rows.length){const h=f>=D.meta.halves[1][1]?1:0,k=Math.floor((f-D.meta.halves[h][1])/FPS),ch=rows[row].s[h][k];t+=' · '+rows[row].name+': '+(row===0?(ch==='V'?'visible':'no visible'):(NAMES[ch]||ch))}showTip(e,t+(f>=W0&&f<W1?' · clic para verlo':''))});
 cv.addEventListener('mouseleave',hideTip);
 cv.addEventListener('click',e=>{const f=at(e);if(f!=null&&f>=W0&&f<W1){seek(f);document.getElementById('pitch').scrollIntoView({behavior:'smooth',block:'center'})}});
 addEventListener('resize',draw);draw();
})();

(function explorer(){
 const W=D.window,R=W.rows,RS=W.reasons,[PL,PW]=D.meta.pitch,step=W.step,nslot=W.slot_team.length;
 const pitch=document.getElementById('pitch'),tl=document.getElementById('timeline'),slider=document.getElementById('slider');
 slider.min=0;slider.max=NW-1;slider.value=0;
 document.getElementById('wlen').textContent=Math.round(NW/FPS/60*10)/10+' min';
 document.getElementById('wclock').textContent=D.meta.window_clock.join(' → ');
 let f=W0,playing=false,speed=1,zoom=false,timer=null;
 const label=(row,fr)=>R[row]?R[row][fr-W0]:null;
 const reason=(row,fr)=>+RS[row][fr-W0];
 function sampleAt(fr){return Math.min(W.pos.length-1,Math.max(0,Math.round((fr-W0)/step)))}
 function find(list,fr){return list.find(g=>fr>=g[0]&&fr<g[1])}
 function lastBall(fr){for(let k=sampleAt(fr);k>=Math.max(0,sampleAt(fr)-25);k--){if(W.ball[k])return [W.ball[k],k]}return null}

 function drawPitch(){
  const [x,w]=sizeCanvas(pitch,Math.round(pitch.clientWidth*(PW+9)/(PL+6))),s=w/(PL+6),ox=3*s,oy=3*s,c=COL();
  const X=v=>ox+v*s,Y=v=>oy+v*s;
  x.fillStyle=css('--pitch');x.fillRect(0,0,w,(PW+9)*s);x.strokeStyle=css('--pline');x.lineWidth=1.2;
  x.strokeRect(X(0),Y(0),PL*s,PW*s);x.beginPath();x.moveTo(X(PL/2),Y(0));x.lineTo(X(PL/2),Y(PW));x.stroke();
  x.beginPath();x.arc(X(PL/2),Y(PW/2),9.15*s,0,7);x.stroke();
  for(const side of [0,1]){const bx=side?PL-16.5:0,gx=side?PL-5.5:0;x.strokeRect(X(bx),Y(PW/2-20.16),16.5*s,40.32*s);x.strokeRect(X(gx),Y(PW/2-9.16),5.5*s,18.32*s);x.strokeRect(X(side?PL:-1.5),Y(PW/2-3.66),1.5*s,7.32*s)}
  const half=D.meta.halves.find(h=>f>=h[1]&&f<=h[2]);
  if(half){const dir=D.meta.direction[half[0]];x.font='12px system-ui';x.fillStyle=c.H;x.fillText(dir>0?'H ataca →':'← H ataca',dir>0?X(0):X(PL-16),Y(PW)+3.6*s);x.fillStyle=c.A;x.fillText(dir>0?'← A ataca':'A ataca →',dir>0?X(PL-16):X(0),Y(PW)+3.6*s)}
  const k=sampleAt(f),pos=W.pos[k];
  const near=W.nearest[k],dist=W.dist[f-W0];
  for(let i=0;i<nslot;i++){const px=pos[2*i],py=pos[2*i+1];if(px==null)continue;x.beginPath();x.arc(X(px/10),Y(py/10),5.2,0,7);x.fillStyle=W.slot_team[i]?c.A:c.H;x.fill();x.lineWidth=1.2;x.strokeStyle=css('--card');x.stroke()}
  const b=W.ball[k];
  if(b){
   if(near>=0&&pos[2*near]!=null){const nx=X(pos[2*near]/10),ny=Y(pos[2*near+1]/10);x.beginPath();x.arc(nx,ny,10,0,7);x.lineWidth=2;x.strokeStyle=css('--ink');x.stroke();
    x.setLineDash([4,3]);x.beginPath();x.moveTo(nx,ny);x.lineTo(X(b[0]/10),Y(b[1]/10));x.lineWidth=1.2;x.stroke();x.setLineDash([]);
    x.font='12px system-ui';x.fillStyle=css('--ink');x.fillText(fmt(dist)+' m',(nx+X(b[0]/10))/2+6,(ny+Y(b[1]/10))/2-6)}
   x.beginPath();x.arc(X(b[0]/10),Y(b[1]/10),4,0,7);x.fillStyle=css('--ink');x.fill();x.lineWidth=1.5;x.strokeStyle=css('--card');x.stroke();
  } else {const lb=lastBall(f);if(lb){x.globalAlpha=.35;x.beginPath();x.arc(X(lb[0][0]/10),Y(lb[0][1]/10),4,0,7);x.fillStyle=css('--ink');x.fill();x.globalAlpha=1}
   x.font='600 13px system-ui';x.fillStyle=css('--bad');x.fillText('Balón no visible',X(1.5),Y(3))}
 }

 function whyText(){
  const r=P.radius_m,out=[],l1=label('s1',f),l2=label('s2',f),l3=label('s3',f),l4=label('s4',f),dist=W.dist[f-W0];
  const r1=reason('s1',f);
  out.push(['Paso 1 · más cercano',l1,r1===0?'No se ve el balón en este frame, así que no hay a quién preguntar: «—».':r1===1?'El jugador más cercano está a '+fmt(dist)+' m (≤ '+fmt(r)+' m) y es del '+(l1==='H'?'Barça':'Athletic')+' → '+l1+'.':'Se ve el balón, pero el más cercano está a '+fmt(dist)+' m (> '+fmt(r)+' m). Probablemente va por el aire o rodando suelto: «—».']);
  const d=find(W.dead,f);let t2;
  if(!d)t2=l2==='D'?'Balón muerto.':'Nada indica que el juego esté parado: se mantiene lo del paso 1.';
  else if(d[2]===1)t2='Todavía no ha habido saque inicial en esta parte ('+d[3]+' en '+clock(d[1])+') → D.';
  else if(d[2]===2)t2='Tramo entre la última vez que el balón se movió rápido dentro del campo ('+clock(d[0])+') y la reanudación que marca el Smart Tagging ('+d[3]+(d[4]!=='?'?', '+d[4]:', equipo N/A')+', '+clock(d[1])+') → D.';
  else if(d[2]===3)t2='El balón se vio fuera del campo en '+clock(d[0])+' y no hay tag de reanudación: D hasta que vuelva a moverse rápido dentro ('+clock(d[1])+').';
  else t2='Los jugadores caminan: su velocidad mediana es '+fmt(W.pace[f-W0])+' m/s (< '+fmt(P.still_speed_ms)+' m/s) desde '+clock(d[0])+' hasta '+clock(d[1])+', y el balón no se mueve rápido → juego parado, D.';
  out.push(['Paso 2 · balón muerto',l2,t2]);
  let t3;const rm=find(W.removed,f);
  if(l3==='D')t3='Balón muerto: no hay rachas que revisar.';
  else if(rm&&reason('s3',f))t3='Aquí había una racha de '+rm[2]+' de solo '+fmt(rm[4])+' s (< '+fmt(P.min_control_s)+' s): un toque sin control, un desvío o ruido del tracking. Se devuelve a '+rm[3]+'.';
  else{const kp=W.kept.find(k=>f>=k[0]&&f<k[1]);t3=kp?'Dentro de una racha de '+kp[2]+' que dura '+fmt((kp[1]-kp[0])/FPS)+' s (≥ '+fmt(P.min_control_s)+' s): es control. '+(l3==='-'?'Este frame sigue en «—» porque no hay observación; lo decide el paso 4.':'Se mantiene.'):(l3==='-'?'Sin observación: lo decide el paso 4.':'Se mantiene.')}
  out.push(['Paso 3 · control',l3,t3]);
  const r4=reason('s4',f),g=find(W.gaps,f);let t4;
  if(l4==='D')t4='Balón muerto (D).';
  else if(r4===0)t4='Ya había una etiqueta observada: no hay nada que rellenar.';
  else if(r4===1)t4='Hueco de '+fmt((g[1]-g[0])/FPS)+' s con '+g[2]+' antes y después → se rellena con '+g[2]+'.';
  else if(r4===2||r4===3)t4='Hueco de '+fmt((g[1]-g[0])/FPS)+' s: antes '+g[2]+', después '+g[3]+'. El cambio se sitúa en '+clock(g[4])+', cuando '+g[3]+' empieza a avanzar. Este frame queda '+(r4===2?'antes':'después')+' del corte → '+l4+'.';
  else if(r4===4)t4='Inicio o final de un tramo en juego sin observación propia: se usa la observación más cercana → '+l4+'.';
  else t4='No hay ninguna observación en todo este tramo en juego → X (el scorer repetirá la última etiqueta).';
  out.push(['Paso 4 · huecos',l4,t4]);
  if(R.truth){const tr=label('truth',f);out.push([D.meta.truth_kind==='full'?'Verdad':'Verdad (pública)',tr,tr==='X'?'Sin verdad en este frame.':(l4===tr?'<span style="color:var(--ok)">✓ El paso 4 acierta.</span>':'<span style="color:var(--bad)">✗ El paso 4 dice '+l4+' y la verdad es '+tr+'.</span>')])}
  document.getElementById('why').innerHTML=out.map(o=>'<div class="k">'+o[0]+'</div><div class="v">'+pill(o[1])+' &nbsp;'+o[2]+'</div>').join('');
 }

 const TROWS=[['ball','Balón visible'],['dist','Distancia (m)'],['pace','Ritmo jugadores'],['s1','Paso 1'],['s2','Paso 2'],['s3','Paso 3'],['s4','Paso 4']].concat(R.truth?[['truth','Verdad'],['err','Errores paso 4']]:[]);
 let tgeo=null;
 function view(){if(!zoom)return [W0,W1];const a=Math.max(W0,Math.min(f-15*FPS,W1-30*FPS));return [a,Math.min(W1,a+30*FPS)]}
 function drawTimeline(){
  const rh=18,g=5,lab=110,dh=40,hgt=TROWS.reduce((s,r)=>s+(r[0]==='dist'||r[0]==='pace'?dh:rh)+g,0)+18;
  const [x,w]=sizeCanvas(tl,hgt),c=COL(),[v0,v1]=view(),n=v1-v0,pw=(w-lab)/n;tgeo={lab,v0,v1,pw};
  let y=0;x.font='12px system-ui';
  for(const [key,name] of TROWS){
   x.fillStyle=css('--mut');x.fillText(name,0,y+13);
   if(key==='dist'||key==='pace'){
    const arr=key==='dist'?W.dist:W.pace,top=key==='dist'?10:6,thr=key==='dist'?P.radius_m:P.still_speed_ms;
    x.fillStyle=css('--soft');x.fillRect(lab,y,w-lab,dh);const sc=v=>y+dh-Math.min(v,top)/top*dh;
    x.strokeStyle=key==='dist'?css('--H'):css('--D');x.setLineDash([3,3]);x.beginPath();x.moveTo(lab,sc(thr));x.lineTo(w,sc(thr));x.stroke();x.setLineDash([]);
    x.strokeStyle=css('--ink');x.lineWidth=1;x.beginPath();let pen=false;
    for(let i=0;i<n;i++){const d=arr[v0-W0+i];if(d==null){pen=false;continue}const px=lab+i*pw,py=sc(d);if(pen)x.lineTo(px,py);else{x.moveTo(px,py);pen=true}}x.stroke();
    x.fillStyle=css('--faint');x.font='10px system-ui';x.fillText(String(top),w-14,y+10);x.fillText(fmt(thr)+(key==='dist'?' m':' m/s'),w-44,sc(thr)-3);x.font='12px system-ui';
    y+=dh+g;continue}
   for(let i=0;i<n;){const fr=v0+i;let ch;
    if(key==='err'){const t=R.truth[fr-W0];ch=(t!=='X'&&t!==R.s4[fr-W0])?'E':'.'}else ch=R[key][fr-W0];
    let j=i+1;while(j<n){const fj=v0+j;const cj=key==='err'?((R.truth[fj-W0]!=='X'&&R.truth[fj-W0]!==R.s4[fj-W0])?'E':'.'):R[key][fj-W0];if(cj!==ch)break;j++}
    x.fillStyle=key==='ball'?(ch==='V'?c.V:c.X):key==='err'?(ch==='E'?css('--bad'):c.X):(c[ch]||c.X);
    x.fillRect(lab+i*pw,y,Math.max(0.6,(j-i)*pw),rh);i=j}
   y+=rh+g}
  x.fillStyle=css('--mut');x.font='11px system-ui';
  const tick=zoom?5:30;for(let s=Math.ceil((v0-W0)/FPS/tick)*tick;W0+s*FPS<v1;s+=tick){const px=lab+(W0+s*FPS-v0)*pw;x.fillText(clock(W0+s*FPS).split(' ')[2],px,y+12)}
  const px=lab+(f-v0)*pw;x.strokeStyle=css('--ink');x.lineWidth=1.5;x.beginPath();x.moveTo(px,0);x.lineTo(px,y);x.stroke();
 }
 function fromX(e){const r=tl.getBoundingClientRect(),px=e.clientX-r.left;if(!tgeo||px<tgeo.lab)return null;return Math.min(tgeo.v1-1,Math.max(tgeo.v0,Math.round(tgeo.v0+(px-tgeo.lab)/tgeo.pw)))}
 tl.addEventListener('click',e=>{const fr=fromX(e);if(fr!=null)go(fr)});
 tl.addEventListener('mousemove',e=>{const fr=fromX(e);if(fr==null)return hideTip();showTip(e,clock(fr)+' · paso 4: '+(NAMES[R.s4[fr-W0]]||'')+(R.truth?' · verdad: '+(NAMES[R.truth[fr-W0]]||''):''))});
 tl.addEventListener('mouseleave',hideTip);
 function go(fr){f=Math.max(W0,Math.min(W1-1,fr));slider.value=f-W0;document.getElementById('clock').textContent=clock(f)+' · frame '+f;drawPitch();whyText();drawTimeline()}
 seek=go;
 slider.addEventListener('input',()=>go(W0+ +slider.value));
 const pb=document.getElementById('play');
 function setPlay(on){playing=on;pb.textContent=on?'❚❚ Pausa':'▶ Reproducir';clearInterval(timer);if(on)timer=setInterval(()=>{if(f>=W1-step){setPlay(false);return}go(f+step*speed)},1000*step/FPS)}
 pb.onclick=()=>setPlay(!playing);
 const sb=document.getElementById('sp');sb.onclick=()=>{speed=speed===1?2:speed===2?4:1;sb.textContent=speed+'×'};
 const zb=document.getElementById('zoom');zb.onclick=()=>{zoom=!zoom;zb.classList.toggle('on',zoom);drawTimeline()};
 addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.key!==' ')return;if(e.key===' '){e.preventDefault();setPlay(!playing)}else if(e.key==='ArrowRight'){e.preventDefault();go(f+1)}else if(e.key==='ArrowLeft'){e.preventDefault();go(f-1)}});
 addEventListener('resize',()=>go(f));
 go(W0);
})();

document.getElementById('steps').innerHTML=D.steps.map((s,i)=>'<div class="card"><div class="sub">Paso '+(i+1)+'</div><h3>'+s.q+'</h3><p style="font-size:14px">'+s.rule+'</p><div>'+s.data.map(d=>'<span class="chip">'+d+'</span>').join('')+'</div><ul>'+s.stats.map(t=>'<li>'+t+'</li>').join('')+'</ul>'+(s.example!=null?'<button style="margin-top:10px" onclick="seek('+s.example+');document.getElementById(\'pitch\').scrollIntoView({behavior:\'smooth\',block:\'center\'})">Ver un ejemplo en el explorador</button>':'<p class="sub" style="margin-top:8px">No hay ejemplos dentro de la ventana del explorador.</p>')+'</div>').join('');

(function errors(){
 const E=D.errors,box=document.getElementById('errsec');
 if(!E){box.innerHTML='';return}
 const maxp=Math.max(...E.by_type.map(t=>t.pct),0.01);
 const o=E.offsets,maxc=Math.max(...o.counts,o.missed,1);
 let hist='<div style="display:flex;align-items:flex-end;gap:3px;height:120px">';
 o.counts.forEach((c,i)=>{const a=o.edges[i];hist+='<div title="'+fmt(a)+' a '+fmt(a+0.5)+' s: '+c+'" style="flex:1;height:'+(c/maxc*100)+'%;background:'+(Math.abs(a+0.25)<=0.5?'var(--ok)':'var(--H)')+';border-radius:3px 3px 0 0"></div>'});
 hist+='<div style="width:10px"></div><div title="no detectados: '+o.missed+'" style="flex:1.4;height:'+(o.missed/maxc*100)+'%;background:var(--bad);border-radius:3px 3px 0 0"></div></div>';
 hist+='<div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--mut);margin-top:4px"><span>5 s antes</span><span>a tiempo</span><span>5 s tarde</span><span style="color:var(--bad)">no detectado</span></div>';
 box.innerHTML='<h2>Dónde falla el paso 4</h2><p class="lead">Comparación frame a frame con la verdad. Aquí está lo que conviene atacar primero.</p><div class="grid2"><div class="card"><h3>Frames mal etiquetados, por tipo</h3><p class="sub">Porcentaje de los frames evaluados.</p>'+E.by_type.map(t=>'<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;font-size:13.5px"><span>'+t.type+'</span><span class="num">'+fmt(t.pct)+'% · '+t.count+' tramos</span></div><div style="height:8px;background:var(--soft);border-radius:4px"><div style="height:8px;width:'+(t.pct/maxp*100)+'%;background:var(--bad);border-radius:4px"></div></div></div>').join('')+'</div><div class="card"><h3>¿Llegan a tiempo los cambios de posesión?</h3><p class="sub">Para cada cambio H↔A de la verdad, la distancia al cambio predicho más cercano. Mediana: '+(o.median==null?'—':fmt(o.median,2)+' s')+'. A ±0,5 s: '+o.within_half_s+' de '+o.n+'.</p>'+hist+'</div></div><div class="card" style="margin-top:12px"><h3>Los errores más largos</h3><table><tr><th>Cuándo</th><th>Duración</th><th>Tipo</th><th>Verdad</th><th>Paso 4</th><th></th></tr>'+E.top.map(s=>'<tr><td class="num">'+s.when+'</td><td class="num">'+fmt(s.seconds)+' s</td><td>'+s.type+'</td><td>'+pill(s.truth)+'</td><td>'+pill(s.pred)+'</td><td>'+(s.start>=W0&&s.start<W1?'<button onclick="seek('+s.start+');document.getElementById(\'pitch\').scrollIntoView({behavior:\'smooth\',block:\'center\'})">Ver</button>':'<span class="sub" title="Vuelve a ejecutar el script con estas opciones para ver este momento en el explorador">fuera de la ventana: <code>'+s.cmd+'</code></span>')+'</td></tr>').join('')+'</table></div>';
})();

(function next(){
 const E=D.errors,top=E&&E.by_type.length?E.by_type[0].type:null;
 const ideas={'D no detectado':'El balón muerto se escapa, sobre todo al principio de cada parada (antes de que los jugadores empiecen a caminar). Prueba a detectar el momento exacto del silbato: frenazo brusco de todos los jugadores, balón fuera aunque no se vea, o la última jugada antes de un tag.','D inventado':'Se marcan paradas que no existen: revisa cómo se elige el inicio del tramo D (último balón rápido) y la regla de balón fuera.','Cambio desplazado':'Los cambios llegan tarde o pronto: es lo que mejor resuelve un decodificador temporal (HMM con Viterbi) en lugar de los pasos 3 y 4.','Posesión inventada o perdida':'Hay posesiones que no existen o que faltan: ajusta la duración mínima de control (paso 3) en el banco de validación con otros partidos.','Sin respuesta':'Quedan tramos sin ninguna observación: usa la forma de los equipos o el Smart Tagging para decidirlos.'};
 let h='<p>Cada paso es deliberadamente simple. Algunas ideas para la siguiente versión:</p><ul>';
 if(top)h+='<li><b>Lo que más pesa ahora: «'+top+'».</b> '+ideas[top]+'</li>';
 h+='<li>Sustituir los pasos 3 y 4 por un HMM con Viterbi: la misma idea («cambiar de equipo cuesta») pero con probabilidades y decidiendo sobre todo el partido a la vez.</li><li>Usar los tags de equipo del Smart Tagging como evidencia adicional en los huecos largos.</li><li>Ajustar los parámetros ('+fmt(P.radius_m)+' m, '+fmt(P.min_control_s)+' s, '+fmt(P.fast_ball_ms,0)+' m/s) con el banco de validación de otros partidos, nunca con los 5 minutos públicos.</li></ul>';
 document.getElementById('next').innerHTML=h;
})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
