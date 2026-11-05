#!/usr/bin/env python3
"""Does the touch detector behave the same on the real tracking? Aggregates only, so it is safe to
run on the real match.

The touches group of src/mtah/features.py finds sharp changes of the ball's velocity, so it depends
on the ball's noise, and the noise in the datasets is partly simulated. This script measures the
same things on the challenge ATD and on the datasets, without labels:

    1. % of graded frames with a ball;
    2. touches per minute of seen ball;
    3. the change of velocity on seen frames, in m/s: median, p75, p90 and p99;
    4. % of touches by the same team as the touch before.

    python tools/touch_rate_diagnostic.py --data-dir "$MTAH_DATA_DIR"     # the real match
    python tools/touch_rate_diagnostic.py --data-dir fake_data/seed0      # the fake match
    python tools/touch_rate_diagnostic.py --datasets external/datasets_v2 # IDSSE and Metrica

REAL DATA: the script prints aggregate numbers only -- no frames, no positions -- and an error
message never includes data. Paste its output, nothing else from the data folder. It writes nothing.
On the datasets, test and test_metrica are left out.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "src", "starter", "evaluation", "tools"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from mtah.features import FPS, detect_touches, velocity_change  # noqa: E402

PERCENTILES = (50, 75, 90, 99)


def measure(players, slot_team, ball, halves):
    """Counts for one match; pooled by summing them, apart from the velocity changes."""
    seen = ~np.isnan(ball[:, 0])
    graded = np.zeros(len(ball), dtype=bool)
    for a, b in halves:
        graded[a:b + 1] = True
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(players - ball[:, None, :], axis=2)
    nearest = []
    for team in (0, 1):
        side = d[:, slot_team == team]
        side = np.where(np.isnan(side), np.inf, side).min(axis=1)
        nearest.append(np.where(seen & np.isfinite(side), side, np.nan).astype(np.float32))
    frames, home = detect_touches(ball, *nearest)
    keep = graded[frames]
    frames, home = frames[keep], home[keep]
    change = velocity_change(ball)[graded & seen]
    return {"graded": int(graded.sum()), "seen": int((graded & seen).sum()), "touches": len(frames),
            "same_team": int(np.sum(home[1:] == home[:-1])), "pairs": max(len(frames) - 1, 0),
            "change": change[~np.isnan(change)]}


def summary(rows):
    graded, seen = sum(r["graded"] for r in rows), sum(r["seen"] for r in rows)
    pct = np.percentile(np.concatenate([r["change"] for r in rows]), PERCENTILES)
    out = {"matches": len(rows), "seen_pct": 100.0 * seen / graded,
           "touches_per_min_seen": sum(r["touches"] for r in rows) / (seen / FPS / 60.0),
           "same_team_pct": 100.0 * sum(r["same_team"] for r in rows) / max(sum(r["pairs"] for r in rows), 1)}
    out.update({"change_p%d_ms" % p: float(v) for p, v in zip(PERCENTILES, pct)})
    return out


def challenge_format(data_dir):
    """The real match, or a fake one in the same layout (tools/make_fake_data.py)."""
    import games
    import heuristic_pipeline as hp
    players, ball, slot_team = hp.load_tracking(games.asset_paths(games.GAME, data_dir), None)
    kickoffs = [games.anchor_video_frame(games.GAMES[games.GAME][k]) for k in ("kickoff", "kickoff_2nd")]
    halves = [(max(first, kick), last) for (_, first, last), kick in zip(hp.HALVES, kickoffs)]
    return [measure(players, np.asarray(slot_team), ball, halves)]


def datasets(dataset_dir):
    """Every match and seed in splits.json but the test ones, by source."""
    import build_datasets as bd
    with open(os.path.join(dataset_dir, "splits.json")) as handle:
        splits = json.load(handle)
    by_source = {}
    for name, items in splits.items():
        if name.startswith("test"):
            continue
        for match, seed in items:
            m = bd.load_match(dataset_dir, match, seed)
            halves = [(a, b) for _, a, b in m["meta"]["halves"]]
            row = measure(m["players"], np.asarray(m["slot_team"]), m["ball"], halves)
            by_source.setdefault(m["meta"]["source"], []).append(row)
    return by_source


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
            columns["data-dir"] = summary(challenge_format(args.data_dir))
        except Exception as error:                                    # noqa: BLE001
            # never echo the exception text: with real data it could contain values from the files
            raise SystemExit("could not read --data-dir (%s); no details printed on purpose" % type(error).__name__)
    if args.datasets:
        columns.update({k: summary(v) for k, v in datasets(args.datasets).items()})

    print("touch detector (aggregates only)")
    names = list(columns)
    print("%-24s %s" % ("", " ".join("%10s" % n[:10] for n in names)))
    for key in next(iter(columns.values())):
        print("%-24s %s" % (key, " ".join("%10.1f" % columns[n][key] for n in names)))


if __name__ == "__main__":
    main()
