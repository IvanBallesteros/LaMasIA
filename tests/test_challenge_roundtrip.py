#!/usr/bin/env python3
"""Round trip: a bank match written in the challenge's format reads back as the dataset holds it.

    python tests/test_challenge_roundtrip.py                          # METRICA1, seed 0, the newest datasets
    python tests/test_challenge_roundtrip.py --match J03WOY --datasets external/datasets

tools/export_challenge_format.py writes the match as the real bundle (ATD .xml + .txt and the
Smart Tagging .xml, under the names games.asset_paths() expects). It is then read with the
readers a submission will use -- there is no submission pipeline yet, so the official starter's
(kloppy through starter/load_atd.py, and starter/load_smart_tagging.py), plus
tools/heuristic_pipeline.load_tracking, which our tools use on the real match -- and compared with
the dataset's npz and XML:

    frames       every ATD record lands on video_frame = frame_id - 1, every graded frame is there
    coordinates  players and ball, back in metres, within the 6 decimals the export writes (< 0.1 mm)
    ball         missing on exactly the same frames
    slots        slot k is "Track k+1", in the same team
    windows      the same Smart Tagging instances as our own parser reads: frames, code and tags

Needs kloppy (pip install -r requirements-starter.txt). About two minutes, most of it kloppy.
Also runs under pytest (test_roundtrip, with the defaults).
"""

import argparse
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "tools", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))

TOLERANCE_M = 1e-4           # 6 decimals of a 0-1 coordinate on a ~105 m pitch: <= 0.053 mm, below the
                             # millimetres the datasets store


def default_datasets():
    for name in ("datasets_v2_1", "datasets_v2", "datasets"):
        if os.path.isdir(os.path.join(ROOT, "external", name)):
            return os.path.join(ROOT, "external", name)
    return os.path.join(ROOT, "external", "datasets")


def _close(a, b):
    """Same NaN pattern and the values within TOLERANCE_M; returns (ok, missing mismatches, max error)."""
    na, nb = np.isnan(a), np.isnan(b)
    both = ~na & ~nb
    err = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    mism = int(np.sum(na != nb))
    return mism == 0 and err <= TOLERANCE_M, mism, err


def check(datasets, match, seed, keep=None):
    import games
    import heuristic_pipeline as hp
    import load_atd
    import load_smart_tagging
    from build_datasets import load_match
    from export_challenge_format import export
    from mtah.train_classifier import set_pieces

    m = load_match(datasets, match, seed)
    meta = m["meta"]
    n, (L, W) = meta["n_frames"], meta["pitch_m"]
    graded = np.zeros(n, dtype=bool)
    for _, a, b in meta["halves"]:
        graded[a:b + 1] = True
    failures, report = [], []

    def expect(ok, what):
        report.append(("ok  " if ok else "FAIL") + "  " + what)
        if not ok:
            failures.append(what)

    work = keep or tempfile.mkdtemp(prefix="roundtrip_", dir=os.path.join(ROOT, "fake_data"))
    t0 = time.time()
    paths = export(datasets, match, seed, work)
    report.append("      exported %s seed %d (%d frames, pitch %g x %g m) to %s in %.0f s" % (
        match, seed, n, L, W, work, time.time() - t0))

    # kloppy, through the official starter
    t0 = time.time()
    dataset = load_atd.load(games.GAME, work)
    df = dataset.to_df()
    ids = df["frame_id"].to_numpy(np.int64)
    video = ids - 1
    expect(len(np.unique(ids)) == len(ids) and video.min() >= 0 and video.max() < n,
           "kloppy: %d records, frame_id %d..%d, all on video frames 0..%d" % (len(ids), ids.min(), ids.max(), n - 1))
    expect(np.all(np.isin(np.flatnonzero(graded), video)), "kloppy: every graded frame has a record")
    order = [p for t in dataset.metadata.teams for p in t.players]
    team_of = {p.player_id: i for i, t in enumerate(dataset.metadata.teams) for p in t.players}
    ok_ids = sorted(int(p.player_id) for p in order) == list(range(1, len(m["slot_team"]) + 1))
    ok_team = ok_ids and all(team_of[str(k + 1)] == int(t) for k, t in enumerate(m["slot_team"]))
    expect(ok_ids and ok_team, "kloppy: slots Track 1..%d, home first, same team as slot_team" % len(m["slot_team"]))
    players = np.stack([df[["%d_x" % (k + 1), "%d_y" % (k + 1)]].to_numpy(np.float64)
                        for k in range(len(m["slot_team"]))], axis=1) * np.array([L, W])
    ball = df[["ball_x", "ball_y"]].to_numpy(np.float64) * np.array([L, W])
    ok, mism, err = _close(players, m["players"][video].astype(np.float64))
    expect(ok, "kloppy: player coordinates (missing mismatches %d, max error %.2g m)" % (mism, err))
    ok, mism, err = _close(ball, m["ball"][video].astype(np.float64))
    expect(ok, "kloppy: ball coordinates (missing mismatches %d, max error %.2g m)" % (mism, err))
    ok = bool(np.array_equal(np.isnan(ball[:, 0]), np.isnan(m["ball"][video, 0])))
    expect(ok, "kloppy: ball missing on the same frames (%.1f%% of graded frames)" % (
        100 * np.mean(np.isnan(m["ball"][graded, 0]))))
    report.append("      kloppy read in %.0f s" % (time.time() - t0))
    del df, dataset

    # tools/heuristic_pipeline.load_tracking: arrays on the challenge's 199319 frames, its pitch
    t0 = time.time()
    hp_players, hp_ball, hp_team = hp.load_tracking(paths, None)
    scale = np.array([L, W]) / np.array(games.GAMES[games.GAME]["pitch"])
    expect(np.array_equal(hp_team, m["slot_team"]), "heuristic_pipeline: slot teams from the ATD metadata")
    ok, mism, err = _close(hp_players[:n] * scale, m["players"].astype(np.float64))
    expect(ok and np.all(np.isnan(hp_players[n:])),
           "heuristic_pipeline: player coordinates (missing mismatches %d, max error %.2g m)" % (mism, err))
    ok, mism, err = _close(hp_ball[:n] * scale, m["ball"].astype(np.float64))
    expect(ok and np.all(np.isnan(hp_ball[n:])),
           "heuristic_pipeline: ball coordinates and missing frames (mismatches %d, max error %.2g m)" % (mism, err))
    report.append("      heuristic_pipeline read in %.0f s" % (time.time() - t0))

    # Smart Tagging: the starter's reader against ours (src/mtah) and the raw dataset XML
    events = load_smart_tagging.load(games.GAME, work)
    ours = set_pieces(m["smart_tagging"], meta["home"], meta["away"])
    theirs = [(e["start_frame"], e["end_frame"],
               {meta["home"]: "H", meta["away"]: "A"}.get((e["tags"].get("Team") or "").strip()))
              for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    expect(theirs == ours, "Smart Tagging: %d SET PIECES windows, same frames and team as src/mtah reads" % len(ours))
    raw = []
    for inst in ET.parse(m["smart_tagging"]).iter("instance"):
        raw.append((inst.findtext("code"), {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}))
    expect([(e["code"], e["tags"]) for e in events] == raw, "Smart Tagging: same codes and tags (Team, Half, Type)")
    lengths = np.array([(hi - lo) / 25.0 for lo, hi, _ in theirs])
    if len(lengths):
        report.append("      SET PIECES window length: min %.2f, median %.2f, max %.2f s" % (
            lengths.min(), np.median(lengths), lengths.max()))

    if not keep:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return failures, report


def test_roundtrip():
    failures, report = check(default_datasets(), "METRICA1", 0)
    assert not failures, "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", default=default_datasets())
    parser.add_argument("--match", default="METRICA1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", default=None, help="export here and keep the files (default: a temporary folder)")
    args = parser.parse_args()
    print("round trip: %s, %s seed %d" % (os.path.relpath(args.datasets, ROOT), args.match, args.seed))
    failures, report = check(args.datasets, args.match, args.seed, args.keep)
    print("\n".join(report))
    print("\n%s" % ("PASSED" if not failures else "FAILED: %d check(s)" % len(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
