"""Per-frame evidence from ATD-like tracking and Smart Tagging. One row per frame, NaN where a
signal has no evidence (the classifier handles missing values itself).

The inputs are the same for the challenge match (tools/heuristic_pipeline.load_tracking) and for
the datasets (tools/build_datasets.load_match): positions in metres from the top-left corner, x
along the pitch, frames on the match's own timeline, slot_team 0 = home. Nothing here reads a
label.

Feature groups, any of which can be left out:

    ball     where the ball is and how fast it moves
    near     who is near the ball right now
    context  who has been near the ball over the last / next seconds (centred windows)
    players  how fast the players move, where each team is and whether it is advancing
    gap      whether the ball is seen, how long it has been / will be missing and the share of
             it seen in centred windows -- DEPENDS ON WHERE THE TRACKER LOSES THE BALL. The
             datasets now copy the real ATD's pattern (tools/ball_gap_diagnostic.py), but only its
             aggregates; keep comparing against a model without this group.
    tags     Smart Tagging SET PIECES windows
    memory   inside a gap: who was near the ball, and where it was, on each side of the gap. NaN
             where the ball is seen. Depends on where the tracker loses the ball, like gap.
    edges    who was near the ball just before and just after this frame, kept apart: a centred
             window reads 0.5 right on a change of possession, these two read 1 and 0
    touches  sharp changes of the ball's velocity next to a player: which team made the last and
             the next one, how many in a row and how many in the last / next seconds
"""

import numpy as np
import pandas as pd

FPS = 25
GROUPS = ("ball", "near", "context", "players", "gap", "tags", "memory", "edges", "touches")
NEAR_M = 3.0                 # "near the ball" for player counts
STILL_MS = 1.5               # a player slower than this is standing
MAX_PLAYER_MS = 12.0         # faster is an identity switch, not running
MAX_BALL_MS = 40.0           # the hardest shots stay below this
CONTEXT_S = (0.5, 2.0, 5.0)  # half-widths of the centred windows
TAG_HORIZON_S = 30.0
GAP_EDGE_S = 1.0             # seconds of seen ball that describe each side of a gap
GAP_EDGE_BALL = ("x_home_attack", "to_touchline", "to_goalline")
TOUCH_K = 8                  # frames on each side to measure the ball's velocity: shorter is tracker noise
TOUCH_MS = 6.0               # smallest change of velocity that counts as a touch
TOUCH_M = 2.5                # how close a player has to be
TOUCH_WINDOW_S = 2.0
COLUMNS = {
    "ball": ["ball_speed", "ball_x_home_attack", "ball_to_touchline", "ball_to_goalline", "ball_outside",
             "ball_speed_1s"],
    "near": ["d_home", "d_away", "d_diff", "n_home_near", "n_away_near", "nearest_home"],
    "context": [c % s for s in CONTEXT_S for c in ("nearest_home_%gs", "d_diff_%gs")],
    "players": ["speed_all", "speed_home", "speed_away", "still_share", "n_objects", "territory_home",
                "territory_away", "advance_home", "advance_away", "speed_all_2s"],
    "gap": ["ball_seen", "gap_since_s", "gap_until_s", "gap_length_s"] + ["ball_seen_%gs" % s for s in CONTEXT_S],
    "tags": ["in_set_piece", "set_piece_team", "to_set_piece_s"],
    "memory": ["mem_last_home", "mem_next_home", "mem_same", "mem_blend"]
              + ["mem_%s_%s" % (side, c) for side in ("last", "next") for c in GAP_EDGE_BALL],
    "edges": [c % s for s in CONTEXT_S for c in ("home_past_%gs", "home_next_%gs", "home_jump_%gs", "d_diff_jump_%gs")],
    "touches": ["tch_last_home", "tch_last_s", "tch_last_run", "tch_next_home", "tch_next_s", "tch_next_run",
                "tch_home_past", "tch_away_past", "tch_home_next", "tch_away_next"],
}


def _speed(xy, k=2):
    """Speed in m/s over +/-k frames, NaN where unknown. xy: (n, 2) or (n, slots, 2)."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) / (2 * k) * FPS
    return out


def _window_mean(values, half):
    """Centred mean over +/-half frames ignoring NaN, and the share of the window with a value."""
    ok = ~np.isnan(values)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0), dtype=np.float64)))
    k = np.concatenate(([0.0], np.cumsum(ok, dtype=np.float64)))
    i = np.arange(len(values))
    lo, hi = np.clip(i - half, 0, len(values)), np.clip(i + half + 1, 0, len(values))
    total, count = s[hi] - s[lo], k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return mean.astype(np.float32), (count / (hi - lo)).astype(np.float32)


def _side_means(values, half):
    """Means ignoring NaN over the `half` frames before and the `half` frames after each frame,
    the frame itself left out."""
    ok = ~np.isnan(values)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0), dtype=np.float64)))
    k = np.concatenate(([0.0], np.cumsum(ok, dtype=np.float64)))
    i = np.arange(len(values))

    def mean(lo, hi):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(k[hi] > k[lo], (s[hi] - s[lo]) / (k[hi] - k[lo]), np.nan).astype(np.float32)

    return mean(np.maximum(i - half, 0), i), mean(i + 1, np.minimum(i + half + 1, len(values)))


def _attack_sign(players, slot_team, halves, L):
    """+1 where the home team attacks towards x = L, -1 towards x = 0 (per half, from where the
    home players stand on average: a team defends its own half)."""
    sign = np.zeros(len(players), dtype=np.float32)
    home = slot_team == 0
    for a, b in halves:
        with np.errstate(invalid="ignore"):
            mean_x = np.nanmean(players[a:b + 1][:, home, 0])
        sign[a:b + 1] = 1.0 if mean_x < L / 2 else -1.0
    return sign


def _gap_lengths(missing):
    """Seconds since the ball was last seen, until it is seen again, and the current gap's length."""
    n = len(missing)
    idx = np.arange(n)
    last = np.where(~missing, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(~missing, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, idx - last, np.nan) / FPS
    until = np.where(nxt < n, nxt - idx, np.nan) / FPS
    return since.astype(np.float32), until.astype(np.float32), (since + until).astype(np.float32)


def _gap_memory(nearest_home, ball, seen, attack, pitch, halves):
    """The memory columns: for each frame without ball, what the seen ball said on each side of its
    gap, within the same half. mem_last_home / mem_next_home are the share of the last / first
    GAP_EDGE_S seconds with a home player nearest; mem_blend moves from one to the other as the gap
    goes by, mem_same is 1 when they agree."""
    L, W = pitch
    n = len(seen)
    half = int(GAP_EDGE_S * FPS) // 2
    centred, _ = _window_mean(nearest_home, half)
    before = np.full(n, np.nan, dtype=np.float32)
    after = np.full(n, np.nan, dtype=np.float32)
    before[2 * half:] = centred[half:n - half]       # frames i - 2 * half ... i
    after[:n - 2 * half] = centred[half:n - half]    # frames i ... i + 2 * half
    x, y = ball[:, 0], ball[:, 1]
    at_edge = {"x_home_attack": (x - L / 2) * attack / (L / 2), "to_touchline": np.minimum(y, W - y),
               "to_goalline": np.minimum(x, L - x)}
    cols = {c: np.full(n, np.nan, dtype=np.float32) for c in COLUMNS["memory"]}
    for a, b in halves:
        idx = np.arange(a, b + 1)
        in_half = seen[a:b + 1]
        last = np.maximum.accumulate(np.where(in_half, idx, -1))
        nxt = np.minimum.accumulate(np.where(in_half, idx, n)[::-1])[::-1]
        for side, ref, values, known in (("last", last, before, last >= 0), ("next", nxt, after, nxt < n)):
            ok = known & ~in_half
            cols["mem_%s_home" % side][idx[ok]] = values[ref[ok]]
            for c in GAP_EDGE_BALL:
                cols["mem_%s_%s" % (side, c)][idx[ok]] = at_edge[c][ref[ok]]
        ok = (last >= 0) & (nxt < n) & ~in_half
        share = (idx[ok] - last[ok]) / (nxt[ok] - last[ok])
        lo, hi = before[last[ok]], after[nxt[ok]]
        cols["mem_blend"][idx[ok]] = lo * (1 - share) + hi * share
        cols["mem_same"][idx[ok]] = 1 - np.abs(hi - lo)
    return cols


def _detect_touches(ball, d_home, d_away):
    """Frames where the ball's velocity changes sharply next to a player, and whether that player
    is a home one."""
    n, k = len(ball), TOUCH_K
    before = np.full(ball.shape, np.nan, dtype=np.float32)
    after = np.full(ball.shape, np.nan, dtype=np.float32)
    before[k:] = (ball[k:] - ball[:-k]) / k * FPS
    after[:-k] = (ball[k:] - ball[:-k]) / k * FPS
    change = np.nan_to_num(np.linalg.norm(after - before, axis=1))
    peak = change >= TOUCH_MS
    for shift in range(1, k + 1):
        peak[shift:] &= change[shift:] >= change[:-shift]
        peak[:-shift] &= change[:-shift] > change[shift:]
    with np.errstate(invalid="ignore"):
        near = np.fmin(d_home, d_away) <= TOUCH_M
        frames = np.flatnonzero(peak & near & ~np.isnan(d_home) & ~np.isnan(d_away))
        return frames, d_home[frames] < d_away[frames]


def _touch_columns(frames, home, n, halves):
    """The touches columns, within each half. A run counts the touches of one team in a row."""
    cols = {c: np.full(n, np.nan, dtype=np.float32) for c in COLUMNS["touches"]}
    window = int(TOUCH_WINDOW_S * FPS)
    for a, b in halves:
        inside = (frames >= a) & (frames <= b)
        f, h = frames[inside], home[inside]
        if not len(f):
            continue
        back, ahead = np.ones(len(f)), np.ones(len(f))
        for j in range(1, len(f)):
            back[j] = back[j - 1] + 1 if h[j] == h[j - 1] else 1
        for j in range(len(f) - 2, -1, -1):
            ahead[j] = ahead[j + 1] + 1 if h[j] == h[j + 1] else 1
        idx = np.arange(a, b + 1)
        last = np.searchsorted(f, idx, side="right") - 1
        nxt = last + 1
        ok = last >= 0
        cols["tch_last_home"][idx[ok]] = h[last[ok]]
        cols["tch_last_s"][idx[ok]] = (idx[ok] - f[last[ok]]) / FPS
        cols["tch_last_run"][idx[ok]] = back[last[ok]]
        ok = nxt < len(f)
        cols["tch_next_home"][idx[ok]] = h[nxt[ok]]
        cols["tch_next_s"][idx[ok]] = (f[nxt[ok]] - idx[ok]) / FPS
        cols["tch_next_run"][idx[ok]] = ahead[nxt[ok]]
        count = np.concatenate(([0], np.cumsum(h)))
        first = np.searchsorted(f, idx - window, side="right")
        end = np.searchsorted(f, idx + window, side="right")
        cols["tch_home_past"][idx] = count[last + 1] - count[first]
        cols["tch_away_past"][idx] = (last + 1 - first) - cols["tch_home_past"][idx]
        cols["tch_home_next"][idx] = count[end] - count[last + 1]
        cols["tch_away_next"][idx] = (end - last - 1) - cols["tch_home_next"][idx]
    return cols


def frame_features(players, slot_team, ball, pitch, halves, set_pieces=(), groups=GROUPS):
    """DataFrame (one row per frame) of the requested feature groups.

    players [n, slots, 2], ball [n, 2] in metres (NaN = missing); slot_team [slots] 0/1;
    pitch (L, W); halves [(first, last)] frames in play; set_pieces [(lo, hi, team)] with team
    'H', 'A' or None (unknown, or N/A as on every throw-in).
    """
    L, W = pitch
    n = len(ball)
    slot_team = np.asarray(slot_team)
    home, away = slot_team == 0, slot_team == 1
    attack = _attack_sign(players, slot_team, halves, L)
    seen = ~np.isnan(ball[:, 0])
    cols = {}

    if "ball" in groups:
        x, y = ball[:, 0], ball[:, 1]
        cols["ball_speed"] = np.minimum(_speed(ball), MAX_BALL_MS)       # jumps are false detections
        cols["ball_x_home_attack"] = (x - L / 2) * attack / (L / 2)
        cols["ball_to_touchline"] = np.minimum(y, W - y)
        cols["ball_to_goalline"] = np.minimum(x, L - x)
        cols["ball_outside"] = np.maximum.reduce([-x, x - L, -y, y - W, np.zeros(n, dtype=np.float32)])
        cols["ball_speed_1s"], _ = _window_mean(cols["ball_speed"], FPS)

    need_near = "near" in groups or "context" in groups or any(g in groups for g in ("memory", "edges", "touches"))
    if need_near:
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(players - ball[:, None, :], axis=2)
        d_home = np.where(np.isnan(d[:, home]), np.inf, d[:, home]).min(axis=1)
        d_away = np.where(np.isnan(d[:, away]), np.inf, d[:, away]).min(axis=1)
        d_home = np.where(seen & np.isfinite(d_home), d_home, np.nan).astype(np.float32)
        d_away = np.where(seen & np.isfinite(d_away), d_away, np.nan).astype(np.float32)
        with np.errstate(invalid="ignore"):
            nearest_home = np.where(np.isnan(d_home) | np.isnan(d_away), np.nan, (d_home < d_away).astype(np.float32))
    if "near" in groups:
        cols["d_home"], cols["d_away"] = d_home, d_away
        cols["d_diff"] = d_home - d_away
        with np.errstate(invalid="ignore"):
            cols["n_home_near"] = np.where(seen, np.sum(d[:, home] < NEAR_M, axis=1), np.nan).astype(np.float32)
            cols["n_away_near"] = np.where(seen, np.sum(d[:, away] < NEAR_M, axis=1), np.nan).astype(np.float32)
        cols["nearest_home"] = nearest_home
    if "context" in groups:
        for s in CONTEXT_S:
            cols["nearest_home_%gs" % s], _ = _window_mean(nearest_home, int(s * FPS))
            cols["d_diff_%gs" % s], _ = _window_mean(d_home - d_away, int(s * FPS))

    if "edges" in groups:
        for s in CONTEXT_S:
            past, following = _side_means(nearest_home, int(s * FPS))
            cols["home_past_%gs" % s], cols["home_next_%gs" % s] = past, following
            cols["home_jump_%gs" % s] = following - past
            past, following = _side_means(d_home - d_away, int(s * FPS))
            cols["d_diff_jump_%gs" % s] = following - past

    if "players" in groups:
        v = _speed(players)
        v = np.where(v > MAX_PLAYER_MS, np.nan, v)
        with np.errstate(invalid="ignore"), _quiet():
            cols["speed_all"] = np.nanmedian(v, axis=1)
            cols["speed_home"] = np.nanmean(v[:, home], axis=1)
            cols["speed_away"] = np.nanmean(v[:, away], axis=1)
            cols["still_share"] = np.nanmean(np.where(np.isnan(v), np.nan, v < STILL_MS), axis=1)
            cx_home = np.nanmean(players[:, home, 0], axis=1)
            cx_away = np.nanmean(players[:, away, 0], axis=1)
        cols["n_objects"] = np.sum(~np.isnan(players[:, :, 0]), axis=1).astype(np.float32)
        cols["territory_home"] = (cx_home - L / 2) * attack / (L / 2)
        cols["territory_away"] = -(cx_away - L / 2) * attack / (L / 2)
        adv_home = np.full(n, np.nan, dtype=np.float32)
        adv_away = np.full(n, np.nan, dtype=np.float32)
        adv_home[FPS:-FPS] = (cx_home[2 * FPS:] - cx_home[:-2 * FPS]) / 2.0 * attack[FPS:-FPS]
        adv_away[FPS:-FPS] = -(cx_away[2 * FPS:] - cx_away[:-2 * FPS]) / 2.0 * attack[FPS:-FPS]
        cols["advance_home"], cols["advance_away"] = adv_home, adv_away
        cols["speed_all_2s"], _ = _window_mean(cols["speed_all"].astype(np.float32), 2 * FPS)

    if "gap" in groups:
        since, until, length = _gap_lengths(~seen)
        cols["ball_seen"] = seen.astype(np.float32)
        cols["gap_since_s"], cols["gap_until_s"], cols["gap_length_s"] = since, until, length
        for s in CONTEXT_S:
            _, cols["ball_seen_%gs" % s] = _window_mean(np.where(seen, 1.0, np.nan).astype(np.float32), int(s * FPS))

    if "tags" in groups:
        inside = np.zeros(n, dtype=np.float32)
        team = np.zeros(n, dtype=np.float32)
        to_start = np.full(n, TAG_HORIZON_S, dtype=np.float32)
        for lo, hi, t in sorted(set_pieces, key=lambda w: w[0]):
            lo, hi = max(int(lo), 0), min(int(hi), n)
            if lo >= n:
                continue
            inside[lo:hi] = 1.0
            team[lo:hi] = {"H": 1.0, "A": -1.0}.get(t, 0.0)
            a, b = max(lo - int(TAG_HORIZON_S * FPS), 0), min(lo + int(TAG_HORIZON_S * FPS), n)
            dist = (np.arange(a, b) - lo) / FPS
            to_start[a:b] = np.where(np.abs(dist) < np.abs(to_start[a:b]), dist, to_start[a:b])
        cols["in_set_piece"], cols["set_piece_team"], cols["to_set_piece_s"] = inside, team, to_start

    if "memory" in groups:
        cols.update(_gap_memory(nearest_home, ball, seen, attack, pitch, halves))

    if "touches" in groups:
        cols.update(_touch_columns(*_detect_touches(ball, d_home, d_away), n, halves))

    assert sorted(cols) == sorted(c for g in groups for c in COLUMNS[g])
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)
