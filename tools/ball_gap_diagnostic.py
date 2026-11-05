#!/usr/bin/env python3
"""Where does the ball go missing? Aggregates only, so it is safe to run on the real match.

The decoder has to know what a stretch without a ball means. In the Metrica datasets the ball is
missing almost only during stoppages; in the IDSSE ones the degradation hides it mostly while the
ball is alive. This script measures the same nine things on the challenge ATD and on the
datasets, so the team can see which pattern the real match follows -- without labels:

    1. % of graded frames without a ball, by the length of the stretch they belong to
       (< 1, 1-2, 2-5, 5-10, 10-30 and > 30 s);
    2. number of stretches without a ball of >= 10 s, per half;
    3. % of Smart Tagging SET PIECES windows preceded by a stretch without a ball of >= 5 s
       (one that ends between 5 s before the window and its end), next to the same % for windows
       of the same (median) length placed at every graded second ("by chance"), and the excess
       over chance, (preceded - chance) / (1 - chance). The chance level grows with the window
       length and with how many long stretches there are, so only the excess compares across
       sources whose windows differ in length; the median window length is printed too;
    4. % of stretches of >= 10 s whose end (the ball seen again) falls within 3 s of a SET PIECES
       window -- of its start, and of anywhere in it -- next to the same % with the windows of each
       half shifted together by a random offset (circularly, 200 draws, fixed seed);
    5. where the ball was last seen before each stretch of >= 10 s: off the pitch, within 2 m of a
       touchline, within 2 m of a goal line, or further inside;
    6. every Smart Tagging window in the file, by code and Type tag: how many, and their length
       (median, p10-p90);
    7. for the stretches of >= 10 s that end between 10 s before the start of a SET PIECES window
       and the end of that window: a histogram of (end of the stretch - start of the window) in
       bins of 2.5 s, next to the counts expected with the windows shifted at random (as in 4). A
       stretch near two overlapping windows counts once for each. If the ball reappears at the
       restart, this is where the restart sits inside the window;
    8. how smooth the tracking is: percentiles of |second difference| (metres per frame^2) of the
       ball over graded frames with a ball in three consecutive frames (the tail, p95-p99.9, holds
       the kicks), the same for the players within each tracklet (a slot seen in three consecutive
       frames), and the % of graded frames with a ball in straight constant-speed stretches
       (|second difference| < LINEAR_EPS_M at >= 3 consecutive frames, so stretches of >= 5
       frames). A tracker that fills short losses by linear interpolation leaves such stretches,
       but so does a smoothed tracker on any ball in flight or rolling: the % only counts
       interpolated frames when the visible ball is noisy, median |second difference| well above
       LINEAR_EPS_M. Percentiles come from a histogram with log bins (~1% wide);
    9. how far the nearest player (either team, any slot) is from a slow ball: percentiles of that
       distance over graded frames with a ball moving slower than SLOW_BALL_MS (central
       difference). A position error of the ball (or of the players) moves the low percentiles: a
       ball at someone's feet stops looking that close. Within one tracker it tells a 3 cm error
       from a 50 cm one (p1 doubles), but the trackers' own conventions move it far more (p50 of
       2.6 m in the clean IDSSE tracking, 0.1 m in Metrica's, which puts the ball on the carrier),
       so on the real ATD it does not measure the error's size.

    python tools/ball_gap_diagnostic.py --data-dir "$MTAH_DATA_DIR"           # the real match
    python tools/ball_gap_diagnostic.py --data-dir fake_data/seed0            # the fake match
    python tools/ball_gap_diagnostic.py --datasets external/datasets          # IDSSE and Metrica
    python tools/ball_gap_diagnostic.py --data-dir ... --datasets ...         # side by side
    python tools/ball_gap_diagnostic.py --datasets external/datasets external/datasets_v2
    python tools/ball_gap_diagnostic.py --datasets external/datasets_v2 --clean   # + truth.npz

REAL DATA: the script prints aggregate numbers only -- no frames, no positions, no single tag, only
counts, shares, length percentiles per code and Type, and percentiles over the whole match of the
second difference of the ball and the players and of the ball's distance to the nearest player --
and an error message never includes
data. --json writes the same aggregates. Paste its output, never anything else from the data folder.
On the challenge match each half starts at its kickoff (games.py anchors): before the second-half
kickoff the ball is dead by definition and would only add one long stretch.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "tools"):
    sys.path.insert(0, os.path.join(ROOT, sub))

FPS = 25
BUCKETS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, None)]
LONG_S = 10.0
PRECEDED_S = 5.0
END_NEAR_S = 3.0
LINE_M = 2.0
SHIFT_DRAWS = 200
WHERE = ("off the pitch", "< %g m from a touchline" % LINE_M, "< %g m from a goal line" % LINE_M, "inside")
OFFSET_BIN_S = 2.5
OFFSET_BEFORE_S = 10.0
LINEAR_EPS_M = 0.10
LINEAR_MIN_INNER = 3
BALL_D2_PERCENTILES = (10, 50, 90, 95, 99, 99.9)
PLAYER_D2_PERCENTILES = (10, 50, 90, 99)
SLOW_BALL_MS = 5.0
NEAREST_PERCENTILES = (1, 5, 10, 25, 50)
D2_EDGES = np.concatenate(([0.0], np.logspace(-5, 3, 1601)))   # metres; the first bin holds exact zeros


# ==================================================================================================
# Measures
# ==================================================================================================

def _runs(mask):
    cut = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.concatenate(([0], cut, [len(mask)]))
    return [(int(p), int(q)) for p, q in zip(bounds[:-1], bounds[1:]) if mask[p]]


def _where(xy, pitch):
    """Index in WHERE of a ball position (metres from a corner)."""
    (x, y), (L, W) = xy, pitch
    if x < 0 or x > L or y < 0 or y > W:
        return 0
    if min(y, W - y) < LINE_M:
        return 1
    if min(x, L - x) < LINE_M:
        return 2
    return 3


def _ends_near(ends, windows, pad, n):
    """How many of the frames in `ends` fall inside some [lo - pad, hi + pad]."""
    cover = np.zeros(n + 1, dtype=np.int64)
    for lo, hi in windows:
        cover[min(max(lo - pad, 0), n)] += 1
        cover[min(max(hi + pad + 1, 0), n)] -= 1
    inside = np.cumsum(cover)[:n] > 0
    return int(np.sum(inside[ends])) if len(ends) else 0


def measure(missing, halves, windows, ball=None, pitch=None, shift_draws=SHIFT_DRAWS):
    """missing: bool per frame; halves: [(first, last)] graded frames; windows: [(lo, hi)] set
    pieces; ball [n, 2] in metres and pitch (L, W), for where the ball was last seen."""
    graded = np.zeros(len(missing), dtype=bool)
    gaps = []                                           # (start, end, half index)
    for h, (a, b) in enumerate(halves):
        graded[a:b + 1] = True
        gaps += [(a + p, a + q, h) for p, q in _runs(missing[a:b + 1])]
    n_graded = int(graded.sum())
    frames_in = {k: 0 for k in range(len(BUCKETS))}
    for p, q, _ in gaps:
        s = (q - p) / FPS
        k = next(i for i, (lo, hi) in enumerate(BUCKETS) if s >= lo and (hi is None or s < hi))
        frames_in[k] += q - p
    long_per_half = [sum(1 for p, q, hh in gaps if hh == h and q - p >= LONG_S * FPS) for h in range(len(halves))]

    ends = np.zeros(len(missing) + 1, dtype=np.int64)  # cumulative count of long-gap ends
    for p, q, _ in gaps:
        if q - p >= PRECEDED_S * FPS:
            ends[q] += 1
    cum = np.concatenate(([0], np.cumsum(ends)))

    def preceded(lo, hi):
        a, b = max(lo - int(PRECEDED_S * FPS), 0), min(hi, len(missing))
        return cum[b + 1] - cum[a] > 0

    inside = [(lo, hi) for lo, hi in windows if 0 <= lo < len(missing) and graded[lo]]
    hit = sum(preceded(lo, hi) for lo, hi in inside)
    width = int(np.median([hi - lo for lo, hi in inside])) if inside else 5 * FPS
    starts = np.flatnonzero(graded)[::FPS]             # one candidate window per graded second
    chance = float(np.mean([preceded(int(f), int(f) + width) for f in starts])) if len(starts) else float("nan")

    # Ends of long stretches near a window (its start / anywhere in it), and with shifted windows.
    near = int(END_NEAR_S * FPS)
    rng = np.random.default_rng(0)
    long_ends, near_start, near_any, shift_start, shift_any = 0, 0, 0, 0.0, 0.0
    offsets, shift_offsets = {}, {}
    where = [0] * len(WHERE)
    for h, (a, b) in enumerate(halves):
        ends = np.array([q - a for p, q, hh in gaps if hh == h and q - p >= LONG_S * FPS and q <= b], dtype=np.int64)
        wins = [(lo - a, hi - a) for lo, hi in inside if a <= lo <= b]
        m = b + 1 - a
        long_ends += len(ends)
        near_start += _ends_near(ends, [(lo, lo) for lo, _ in wins], near, m)
        near_any += _ends_near(ends, wins, near, m)
        _offset_bins(ends, wins, offsets, 1.0)
        for _ in range(shift_draws):
            k = int(rng.integers(m))
            moved = [((lo + k) % m, (lo + k) % m + hi - lo) for lo, hi in wins]
            split = [w for lo, hi in moved for w in ([(lo, hi)] if hi < m else [(lo, m - 1), (0, hi - m)])]
            shift_start += _ends_near(ends, [(lo, lo) for lo, _ in moved], near, m) / shift_draws
            shift_any += _ends_near(ends, split, near, m) / shift_draws
            _offset_bins(ends, [(lo, hi) for lo, hi in moved if hi < m], shift_offsets, 1.0 / shift_draws)
        if ball is not None:
            for p, q, hh in gaps:
                if hh == h and q - p >= LONG_S * FPS and p > a:
                    where[_where(ball[p - 1], pitch)] += 1
    return {"graded_frames": n_graded, "missing": float(missing[graded].mean()),
            "by_length": [frames_in[k] / n_graded for k in range(len(BUCKETS))],
            "long_per_half": long_per_half, "set_pieces": len(inside),
            "preceded": hit / len(inside) if inside else float("nan"), "chance": chance,
            "window_lengths": [(hi - lo) / FPS for lo, hi in inside], "long_ends": long_ends,
            "near_start": near_start, "near_any": near_any, "shift_start": shift_start, "shift_any": shift_any,
            "where": where, "offsets": offsets, "shift_offsets": shift_offsets}


def _offset_bins(ends, windows, counts, weight):
    """Add `weight` to counts[bin] for every (end, window) with lo - OFFSET_BEFORE_S <= end <= hi,
    the bin of (end - lo) in OFFSET_BIN_S steps (bin k covers [k, k + 1) * OFFSET_BIN_S seconds)."""
    if not len(ends):
        return
    ends = np.sort(ends)
    before = int(OFFSET_BEFORE_S * FPS)
    for lo, hi in windows:
        sel = ends[np.searchsorted(ends, lo - before):np.searchsorted(ends, hi, side="right")]
        for k in np.floor((sel - lo) / FPS / OFFSET_BIN_S).astype(int).tolist():
            counts[k] = counts.get(k, 0.0) + weight


def _d2_hist(values):
    return np.histogram(np.clip(values, 0.0, D2_EDGES[-1]), D2_EDGES)[0].astype(np.int64)


def d2_percentile(hist, q):
    """Percentile q of the values binned in hist (D2_EDGES): the bin's geometric centre."""
    cum = np.cumsum(hist)
    if not len(cum) or cum[-1] == 0:
        return float("nan")
    k = int(np.searchsorted(cum, q / 100.0 * cum[-1]))
    lo, hi = D2_EDGES[k], D2_EDGES[k + 1]
    return float(np.sqrt(lo * hi)) if lo > 0 else 0.0


def smoothness(ball, halves, players=None):
    """Measure 8 on ball [n, 2] and players [n, slots, 2] in metres (NaN = missing): {"d2_hist":
    histogram of the ball's |second difference| at graded frames with a ball in three consecutive
    frames, "player_d2_hist": the same per slot, "linear": graded frames with a ball in straight
    constant-speed stretches, "ball_frames": graded frames with a ball}."""
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for a, b in halves:
        graded[a:b + 1] = True
    d2 = np.full(n, np.nan)
    d2[1:-1] = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=1)
    valid = graded & np.isfinite(d2)
    small = np.zeros(n, dtype=bool)
    for p, q in _runs(valid & (np.nan_to_num(d2, nan=np.inf) < LINEAR_EPS_M)):
        if q - p >= LINEAR_MIN_INNER:
            small[max(p - 1, 0):min(q + 1, n)] = True          # the inner frames and their neighbours
    has_ball = graded & ~np.isnan(ball[:, 0])
    out = {"d2_hist": _d2_hist(d2[valid]), "linear": int(np.sum(small & has_ball)), "ball_frames": int(has_ball.sum())}
    if players is not None:
        hist = np.zeros(len(D2_EDGES) - 1, dtype=np.int64)
        for s in range(1, n - 1, 20000):
            e = min(s + 20000, n - 1)
            block = players[s - 1:e + 1].astype(np.float64)
            d = np.linalg.norm(block[2:] - 2 * block[1:-1] + block[:-2], axis=2)
            hist += _d2_hist(d[graded[s:e, None] & np.isfinite(d)])
        out["player_d2_hist"] = hist
        out["nearest_hist"] = nearest_player(ball, players, graded)
    return out


def nearest_player(ball, players, graded):
    """Measure 9: histogram (D2_EDGES bins, metres) of the distance from the ball to the nearest
    player at graded frames with a ball slower than SLOW_BALL_MS."""
    n = len(ball)
    speed = np.full(n, np.nan)
    speed[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) / 2.0 * FPS
    idx = np.flatnonzero(graded & (np.nan_to_num(speed, nan=np.inf) < SLOW_BALL_MS))
    hist = np.zeros(len(D2_EDGES) - 1, dtype=np.int64)
    for s in range(0, len(idx), 20000):
        f = idx[s:s + 20000]
        d = np.linalg.norm(players[f].astype(np.float64) - ball[f, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d).min(axis=1)
        hist += _d2_hist(d[np.isfinite(d)])
    return hist


def tag_lengths(instances):
    """{(code, Type): [length in seconds]} for every instance of a Smart Tagging file;
    instances: [(code, Type or None, start_frame, end_frame)]."""
    out = {}
    for code, kind, lo, hi in instances:
        out.setdefault(((code or "?").strip().upper(), (kind or "-").strip()), []).append((hi - lo) / FPS)
    return out


def _sum_bins(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


def pool(results):
    """Several matches (or seeds) of one source -> one column, frame-weighted."""
    w = np.array([r["graded_frames"] for r in results], dtype=float)
    sp = np.array([r["set_pieces"] for r in results], dtype=float)
    halves = [n for r in results for n in r["long_per_half"]]
    return {"n": len(results), "missing": float(np.average([r["missing"] for r in results], weights=w)),
            "by_length": list(np.average([r["by_length"] for r in results], axis=0, weights=w)),
            "long_per_half": (float(np.mean(halves)), int(min(halves)), int(max(halves))),
            "set_pieces": int(sp.sum()),
            "preceded": float(np.average([r["preceded"] for r in results], weights=sp)) if sp.sum() else float("nan"),
            "chance": float(np.average([r["chance"] for r in results], weights=w)),
            "window_lengths": [x for r in results for x in r["window_lengths"]],
            "where": [sum(r["where"][i] for r in results) for i in range(len(WHERE))],
            "offsets": _sum_bins(r["offsets"] for r in results),
            "shift_offsets": _sum_bins(r["shift_offsets"] for r in results),
            "tags": {k: [x for r in results for x in r.get("tags", {}).get(k, [])]
                     for k in sorted({k for r in results for k in r.get("tags", {})})},
            **{k: sum((r[k] for r in results if k in r), np.zeros(len(D2_EDGES) - 1, dtype=np.int64))
               for k in ("d2_hist", "player_d2_hist", "nearest_hist")},
            "linear": sum(r.get("linear", 0) for r in results), "ball_frames": sum(r.get("ball_frames", 0) for r in results),
            **{k: sum(r[k] for r in results) for k in ("long_ends", "near_start", "near_any", "shift_start", "shift_any")}}


# ==================================================================================================
# Sources
# ==================================================================================================

def challenge_format(data_dir):
    """The real match, or a fake one in the same layout (tools/make_fake_data.py)."""
    import games
    import heuristic_pipeline as hp
    import load_smart_tagging
    paths = games.asset_paths(games.GAME, data_dir)
    players, ball, _ = hp.load_tracking(paths, None)
    pitch = tuple(games.GAMES[games.GAME]["pitch"])
    kickoffs = [games.anchor_video_frame(games.GAMES[games.GAME][k]) for k in ("kickoff", "kickoff_2nd")]
    halves = [(max(first, kick), last) for (_, first, last), kick in zip(hp.HALVES, kickoffs)]
    events = load_smart_tagging.load(games.GAME, data_dir)
    windows = [(e["start_frame"], e["end_frame"]) for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    result = measure(np.isnan(ball[:, 0]), halves, windows, ball, pitch)
    result["tags"] = tag_lengths([(e["code"], e["tags"].get("Type"), e["start_frame"], e["end_frame"]) for e in events])
    result.update(smoothness(ball.astype(np.float64), halves, players))
    return pool([result])


def _from_cm(a):
    out = a.astype(np.float32) / 100.0
    out[a == -32768] = np.nan
    return out


def _from_mm(a):
    out = a.astype(np.float64) / 1000.0
    out[a == -2 ** 31] = np.nan
    return out


def datasets(dataset_dir, clean=False):
    """Every match and seed in splits.json, pooled per source. clean: each match's clean tracking
    (truth.npz, centimetres) once instead, with its first seed's Smart Tagging."""
    with open(os.path.join(dataset_dir, "splits.json")) as handle:
        splits = json.load(handle)
    by_source, done = {}, set()
    for items in splits.values():
        for match, seed in items:
            if clean and match in done:
                continue
            done.add(match)
            folder = os.path.join(dataset_dir, "matches", match)
            with open(os.path.join(folder, "meta.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
            if clean:
                npz = np.load(os.path.join(folder, "truth.npz"))
                ball, players = _from_cm(npz["ball_cm"][:, :2]), _from_cm(npz["players_cm"])
            else:
                npz = np.load(os.path.join(folder, "seed%d" % seed, "tracking.npz"))
                if "ball_mm" in npz.files:                          # v2.1 on: millimetres
                    ball, players = _from_mm(npz["ball_mm"]), _from_mm(npz["slots_mm"])
                else:
                    ball, players = _from_cm(npz["ball_cm"]), _from_cm(npz["slots_cm"])
            instances = []
            for inst in ET.parse(os.path.join(folder, "seed%d" % seed, "smart_tagging.xml")).iter("instance"):
                kind = next((lab.findtext("text") for lab in inst.findall("label") if lab.findtext("group") == "Type"), None)
                instances.append((inst.findtext("code"), kind, int(round(float(inst.findtext("start")) * FPS)),
                                  int(round(float(inst.findtext("end")) * FPS))))
            windows = [(lo, hi) for code, _, lo, hi in instances if (code or "").strip().lower() == "set pieces"]
            halves = [(a, b) for _, a, b in meta["halves"]]
            result = measure(np.isnan(ball[:, 0]), halves, windows, ball, tuple(meta["pitch_m"]))
            result["tags"] = tag_lengths(instances)
            result.update(smoothness(ball.astype(np.float64), halves, players))
            by_source.setdefault(meta["source"] + (" clean" if clean else ""), []).append(result)
    return {source: pool(rs) for source, rs in sorted(by_source.items())}


# ==================================================================================================
# Output
# ==================================================================================================

def table(columns):
    names = list(columns)
    width = max(17, *(len(n) for n in names))

    def line(label, cells):
        print("  %-38s %s" % (label, "  ".join("%*s" % (width, c) for c in cells)))

    line("", names)
    line("matches (x seeds) pooled", ["%d" % columns[n]["n"] for n in names])
    line("graded frames without a ball", ["%.1f%%" % (100 * columns[n]["missing"]) for n in names])
    for k, (lo, hi) in enumerate(BUCKETS):
        label = "  in stretches of %s s" % ("%g-%g" % (lo, hi) if hi else "> %g" % lo)
        if lo == 0:
            label = "  in stretches of < %g s" % hi
        line(label, ["%.1f%%" % (100 * columns[n]["by_length"][k]) for n in names])
    line("stretches >= %g s per half (min-max)" % LONG_S,
         ["%.1f (%d-%d)" % columns[n]["long_per_half"] for n in names])
    line("SET PIECES windows", ["%d" % columns[n]["set_pieces"] for n in names])
    line("  preceded by a gap >= %g s" % PRECEDED_S, ["%.1f%%" % (100 * columns[n]["preceded"]) for n in names])
    line("  same, random windows (chance)", ["%.1f%%" % (100 * columns[n]["chance"]) for n in names])
    line("  excess over chance", ["%.1f%%" % (100 * (columns[n]["preceded"] - columns[n]["chance"]) /
                                               (1 - columns[n]["chance"])) for n in names])
    line("  window length, median (p10-p90)", ["%.0f s (%.0f-%.0f)" % (np.median(columns[n]["window_lengths"]),
                                               *np.percentile(columns[n]["window_lengths"], (10, 90)))
                                               if columns[n]["window_lengths"] else "-" for n in names])
    line("stretches >= %g s that end in a half" % LONG_S, ["%d" % columns[n]["long_ends"] for n in names])
    for key, label in (("start", "end within %g s of a window start" % END_NEAR_S),
                       ("any", "end within %g s of a window" % END_NEAR_S)):
        line("  " + label, ["%.1f%%" % (100 * columns[n]["near_" + key] / max(columns[n]["long_ends"], 1)) for n in names])
        line("    same, windows shifted at random", ["%.1f%%" % (100 * columns[n]["shift_" + key] /
                                                      max(columns[n]["long_ends"], 1)) for n in names])
    line("ball last seen before a stretch >= %g s" % LONG_S, ["%d" % sum(columns[n]["where"]) for n in names])
    for i, label in enumerate(WHERE):
        line("  " + label, ["%.1f%%" % (100 * columns[n]["where"][i] / max(sum(columns[n]["where"]), 1)) for n in names])

    print("\n  smoothness: |second difference| in metres, graded frames (8)")
    for who, key, qs in (("ball", "d2_hist", BALL_D2_PERCENTILES), ("players, per tracklet", "player_d2_hist",
                                                                    PLAYER_D2_PERCENTILES)):
        line("  " + who, [""] * len(names))
        for q in qs:
            line("    p%g" % q, ["%.4f" % d2_percentile(columns[n][key], q) if columns[n][key].sum() else "-" for n in names])
    line("  nearest player to a ball < %g m/s (9)" % SLOW_BALL_MS, [""] * len(names))
    for q in NEAREST_PERCENTILES:
        line("    p%g" % q, ["%.3f" % d2_percentile(columns[n]["nearest_hist"], q) if columns[n]["nearest_hist"].sum() else "-"
                             for n in names])
    line("  ball in straight constant-speed stretches", ["%.1f%%" % (100 * columns[n]["linear"] / max(columns[n]["ball_frames"], 1))
                                                         for n in names])

    print("\n  Smart Tagging windows by code / Type (whole file): n, length median (p10-p90; not below 5 windows)")
    for key in sorted({k for n in names for k in columns[n]["tags"]}):
        line("  %s / %s" % key, ["-" if not v else "%d, %.0f s" % (len(v), np.median(v)) if len(v) < 5 else
                                 "%d, %.0f s (%.0f-%.0f)" % (len(v), np.median(v), *np.percentile(v, (10, 90)))
                                 for v in (columns[n]["tags"].get(key, []) for n in names)])

    print("\n  stretches >= %g s ending from %g s before a SET PIECES window start to its end:" % (LONG_S, OFFSET_BEFORE_S))
    print("  end - window start, bins of %g s: count / expected with windows shifted at random" % OFFSET_BIN_S)
    bins = sorted({k for n in names for k in list(columns[n]["offsets"]) + list(columns[n]["shift_offsets"])})
    line("  (stretch, window) pairs", ["%d / %.1f" % (sum(columns[n]["offsets"].values()),
                                                     sum(columns[n]["shift_offsets"].values())) for n in names])
    for k in bins:
        line("  [%+5.1f, %+5.1f) s" % (k * OFFSET_BIN_S, (k + 1) * OFFSET_BIN_S),
             ["%d / %.1f" % (columns[n]["offsets"].get(k, 0), columns[n]["shift_offsets"].get(k, 0.0)) for n in names])


def summary(column):
    """The printed numbers of one column, without per-window lists (for --json)."""
    c, ends = column, max(column["long_ends"], 1)
    return {"matches": c["n"], "missing": c["missing"], "by_length": c["by_length"],
            "long_per_half": c["long_per_half"][0], "set_pieces": c["set_pieces"], "preceded": c["preceded"],
            "chance": c["chance"], "excess": (c["preceded"] - c["chance"]) / (1 - c["chance"]),
            "window_median_s": float(np.median(c["window_lengths"])) if c["window_lengths"] else None,
            "long_ends": c["long_ends"], "near_start": c["near_start"] / ends, "near_any": c["near_any"] / ends,
            "shift_start": c["shift_start"] / ends, "shift_any": c["shift_any"] / ends,
            "where": [x / max(sum(c["where"]), 1) for x in c["where"]],
            "tags": {"%s / %s" % k: {"n": len(v), "median_s": float(np.median(v))} for k, v in c["tags"].items()},
            "ball_d2": {"p%g" % q: d2_percentile(c["d2_hist"], q) for q in BALL_D2_PERCENTILES},
            "player_d2": {"p%g" % q: d2_percentile(c["player_d2_hist"], q) for q in PLAYER_D2_PERCENTILES},
            "nearest_player_slow_ball": {"p%g" % q: d2_percentile(c["nearest_hist"], q) for q in NEAREST_PERCENTILES},
            "linear": c["linear"] / max(c["ball_frames"], 1), "linear_eps_m": LINEAR_EPS_M,
            "offset_bin_s": OFFSET_BIN_S, "offsets": {str(k): v for k, v in sorted(c["offsets"].items())},
            "shift_offsets": {str(k): round(v, 2) for k, v in sorted(c["shift_offsets"].items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="challenge-format data (the real match, or fake_data/seed0)")
    parser.add_argument("--datasets", nargs="+", default=None, help="folder(s) written by tools/build_datasets.py")
    parser.add_argument("--clean", action="store_true", help="also measure each --datasets folder's clean tracking")
    parser.add_argument("--json", default=None, help="also write the printed aggregates here")
    args = parser.parse_args()
    if not args.data_dir and not args.datasets:
        parser.error("give --data-dir, --datasets or both")

    columns = {}
    if args.data_dir:
        try:
            columns["data-dir"] = challenge_format(args.data_dir)
        except Exception as error:                                    # noqa: BLE001
            # never echo the exception text: with real data it could contain values from the files
            raise SystemExit("could not read --data-dir (%s); no details printed on purpose" % type(error).__name__)
    for folder in args.datasets or []:
        prefix = os.path.basename(os.path.normpath(folder)) + ":" if len(args.datasets) > 1 else ""
        columns.update({prefix + k: v for k, v in datasets(folder).items()})
        if args.clean:
            columns.update({prefix + k: v for k, v in datasets(folder, clean=True).items()})
    print("ball gaps (aggregates only)")
    table(columns)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({k: summary(v) for k, v in columns.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
