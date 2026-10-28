#!/usr/bin/env python3
"""Where does the ball go missing? Aggregates only, so it is safe to run on the real match.

The decoder has to know what a stretch without a ball means. In the Metrica datasets the ball is
missing almost only during stoppages; in the IDSSE ones the degradation hides it mostly while the
ball is alive. This script measures the same five things on the challenge ATD and on the
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
       touchline, within 2 m of a goal line, or further inside.

    python tools/ball_gap_diagnostic.py --data-dir "$MTAH_DATA_DIR"           # the real match
    python tools/ball_gap_diagnostic.py --data-dir fake_data/seed0            # the fake match
    python tools/ball_gap_diagnostic.py --datasets external/datasets          # IDSSE and Metrica
    python tools/ball_gap_diagnostic.py --data-dir ... --datasets ...         # side by side

REAL DATA: the script prints aggregate numbers only -- no frames, no positions, no tags -- and an
error message never includes data. Paste its output, never anything else from the data folder.
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
    where = [0] * len(WHERE)
    for h, (a, b) in enumerate(halves):
        ends = np.array([q - a for p, q, hh in gaps if hh == h and q - p >= LONG_S * FPS and q <= b], dtype=np.int64)
        wins = [(lo - a, hi - a) for lo, hi in inside if a <= lo <= b]
        m = b + 1 - a
        long_ends += len(ends)
        near_start += _ends_near(ends, [(lo, lo) for lo, _ in wins], near, m)
        near_any += _ends_near(ends, wins, near, m)
        for _ in range(shift_draws):
            k = int(rng.integers(m))
            moved = [((lo + k) % m, (lo + k) % m + hi - lo) for lo, hi in wins]
            split = [w for lo, hi in moved for w in ([(lo, hi)] if hi < m else [(lo, m - 1), (0, hi - m)])]
            shift_start += _ends_near(ends, [(lo, lo) for lo, _ in moved], near, m) / shift_draws
            shift_any += _ends_near(ends, split, near, m) / shift_draws
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
            "where": where}


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
    _, ball, _ = hp.load_tracking(paths, None)
    pitch = tuple(games.GAMES[games.GAME]["pitch"])
    kickoffs = [games.anchor_video_frame(games.GAMES[games.GAME][k]) for k in ("kickoff", "kickoff_2nd")]
    halves = [(max(first, kick), last) for (_, first, last), kick in zip(hp.HALVES, kickoffs)]
    windows = [(e["start_frame"], e["end_frame"]) for e in load_smart_tagging.load(games.GAME, data_dir)
               if (e["code"] or "").strip().lower() == "set pieces"]
    return pool([measure(np.isnan(ball[:, 0]), halves, windows, ball, pitch)])


def datasets(dataset_dir):
    """Every match and seed in splits.json, pooled per source."""
    with open(os.path.join(dataset_dir, "splits.json")) as handle:
        splits = json.load(handle)
    by_source = {}
    for items in splits.values():
        for match, seed in items:
            folder = os.path.join(dataset_dir, "matches", match)
            with open(os.path.join(folder, "meta.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
            ball_cm = np.load(os.path.join(folder, "seed%d" % seed, "tracking.npz"))["ball_cm"]
            ball = np.where(ball_cm == -32768, np.nan, ball_cm / 100.0)
            windows = []
            for inst in ET.parse(os.path.join(folder, "seed%d" % seed, "smart_tagging.xml")).iter("instance"):
                if (inst.findtext("code") or "").strip().lower() == "set pieces":
                    windows.append((int(round(float(inst.findtext("start")) * FPS)),
                                    int(round(float(inst.findtext("end")) * FPS))))
            halves = [(a, b) for _, a, b in meta["halves"]]
            by_source.setdefault(meta["source"], []).append(
                measure(np.isnan(ball[:, 0]), halves, windows, ball, tuple(meta["pitch_m"])))
    return {source: pool(rs) for source, rs in sorted(by_source.items())}


# ==================================================================================================
# Output
# ==================================================================================================

def table(columns):
    names = list(columns)
    width = max(12, *(len(n) for n in names))

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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="challenge-format data (the real match, or fake_data/seed0)")
    parser.add_argument("--datasets", default=None, help="a folder written by tools/build_datasets.py")
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
    if args.datasets:
        columns.update(datasets(args.datasets))
    print("ball gaps (aggregates only)")
    table(columns)


if __name__ == "__main__":
    main()
