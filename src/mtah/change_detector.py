#!/usr/bin/env python3
"""Turnover detector: per frame, the probability that the ball changes team (H <-> A with no stoppage
in between) within --radius-s, for an input-dependent switch cost in the decoder.

    python src/mtah/change_detector.py
    python src/mtah/decoder.py --change external/predictions_v2_1/change_gbm_harmonized_r0.4

Why: most of the possessions the decoder misses are short ones that start with a turnover in play
(src/mtah/decoder_analysis.py), and the decoder pays the same switch cost everywhere. A signal that
says where a turnover is likely lets it switch cheaply there and dearly elsewhere.

Same protocol as train_classifier.py: each round leaves one IDSSE match out (all its seeds), trains
on the other five and predicts the held-out match and val_metrica; val_metrica gets the mean of the
six rounds. Same model and hyperparameters (train_classifier.HYPERPARAMETERS) on the features of
src/mtah/features.py (default: every group but touches, so the classifier's plus turn, the ball's
direction before and after). The target comes from the labels of the public datasets (--labels for
IDSSE, Metrica's own for val_metrica). test and test_metrica are never read; nothing comes from the
challenge match.

Saved under --out/<name>/:

    oof/<match>_s<seed>.npz          p_change of the held-out match (float32, every frame)
    val_metrica/<game>_s<seed>.npz   the mean of the six rounds
    summary.json                     the base rate the decoder centres on, and per held-out match and
                                     on val_metrica: ROC AUC, average precision and the share of true
                                     changes the detector catches in its top 10% of frames
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for sub in ("src", "tools", "evaluation", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from mtah.features import COLUMNS, GROUPS  # noqa: E402
from mtah.train_classifier import CLASSES, CV_SPLITS, FPS, HYPERPARAMETERS, load, resolve_datasets  # noqa: E402

DEFAULT_GROUPS = tuple(g for g in GROUPS if g != "touches")
TOP = 90            # percentile: "caught" means the detector is in its top 10% of frames at the change


def direct_changes(labels):
    """Frames where a new team starts straight after the other one (H <-> A, no D in between)."""
    team = np.isin(labels, ("H", "A"))
    return np.flatnonzero(team[1:] & team[:-1] & (labels[1:] != labels[:-1])) + 1


def near_change(labels, radius):
    """True within `radius` frames of a direct change."""
    y = np.zeros(len(labels), dtype=bool)
    for c in direct_changes(labels):
        y[max(c - radius, 0):c + radius + 1] = True
    return y


def detection(p, labels, scored, radius):
    """How well p_change points at the true changes, on the scored frames."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = near_change(labels, radius)[scored]
    q = p[scored]
    threshold = np.percentile(q, TOP)
    changes = direct_changes(labels)
    caught = [p[max(c - radius, 0):c + radius + 1].max() >= threshold for c in changes]
    return {"auc": float(roc_auc_score(y, q)), "average_precision": float(average_precision_score(y, q)),
            "positive_rate": float(y.mean()), "changes": int(len(changes)),
            "caught_in_top10": float(np.mean(caught)) if caught else None}


def mean_rows(rows, keys=("auc", "average_precision", "positive_rate", "caught_in_top10")):
    """Mean over seeds within a match, then over matches (every match weighs the same)."""
    matches = sorted({r["match"] for r in rows})
    per = [{k: float(np.mean([r[k] for r in rows if r["match"] == m])) for k in keys} for m in matches]
    return {k: float(np.mean([r[k] for r in per])) for k in keys}


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", default=None, help="default: the current datasets (v2.1)")
    parser.add_argument("--labels", choices=("harmonized", "raw"), default="harmonized",
                        help="IDSSE labels the changes are taken from (default: harmonized, the challenge's notion "
                             "of control)")
    parser.add_argument("--radius-s", type=float, default=0.4, help="a frame is positive within this of a change")
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS), choices=GROUPS,
                        help="feature groups (default: every group but touches)")
    parser.add_argument("--step", type=int, default=5, help="train on one frame in STEP (per match)")
    parser.add_argument("--tag", default="", help="added to the run's name")
    parser.add_argument("--out", default=None, help="default: external/predictions + the datasets suffix")
    args = parser.parse_args()
    args.datasets = resolve_datasets(args.datasets)
    args.out = args.out or os.path.join(
        ROOT, "external", os.path.basename(os.path.normpath(args.datasets)).replace("datasets", "predictions", 1))
    groups = tuple(g for g in GROUPS if g in args.groups)
    cols = [c for g in groups for c in COLUMNS[g]]
    radius = int(round(args.radius_s * FPS))
    name = "change_gbm_%s_r%g%s" % (args.labels, args.radius_s, "_" + args.tag if args.tag else "")
    out = os.path.join(args.out, name)
    for sub in ("oof", "val_metrica"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    print("datasets %s -> %s\nfeature groups: %s" % (args.datasets, out, " ".join(groups)))

    with open(os.path.join(args.datasets, "splits.json")) as handle:
        splits = json.load(handle)
    pool = [tuple(x) for split in CV_SPLITS for x in splits[split]]
    metrica = [tuple(x) for x in splits.get("val_metrica", [])]
    matches = sorted({m for m, _ in pool})
    t0 = time.time()
    data = {x: load(args.datasets, *x, groups=groups) for x in pool + metrica}
    labels = {x: data[x]["labels"][args.labels if x in pool else "harmonized"] for x in data}
    scored = {x: data[x]["graded"] & np.isin(labels[x], CLASSES) for x in data}
    target = {x: near_change(labels[x], radius) for x in data}
    print("%d matches x seeds loaded (%.0f s)" % (len(data), time.time() - t0))

    base = float(np.mean(np.concatenate([target[x][scored[x]] for x in pool])))
    oof_rows, ensemble = [], {}
    for held in matches:
        xs, ys = [], []
        for match in matches:
            if match == held:
                continue
            seeds = [s for m, s in pool if m == match]
            for seed in seeds:                          # every match weighs the same, whatever its seeds
                keep = np.flatnonzero(scored[match, seed])[::args.step * len(seeds)]
                xs.append(data[match, seed]["features"].iloc[keep][cols])
                ys.append(target[match, seed][keep])
        model = HistGradientBoostingClassifier(**HYPERPARAMETERS)
        model.fit(pd.concat(xs, ignore_index=True), np.concatenate(ys))
        positive = list(model.classes_).index(True)
        for x in [x for x in pool if x[0] == held] + metrica:
            p = model.predict_proba(data[x]["features"][cols])[:, positive].astype(np.float32)
            if x[0] == held:
                np.savez_compressed(os.path.join(out, "oof", "%s_s%d.npz" % x), p_change=p)
                row = dict(detection(p, labels[x], scored[x], radius), match=x[0], seed=x[1])
                oof_rows.append(row)
                print("  held out %-8s s%d   AUC %.3f   average precision %.3f (base %.3f)   caught in top 10%% %.2f" % (
                    x[0], x[1], row["auc"], row["average_precision"], row["positive_rate"], row["caught_in_top10"]))
            else:
                ensemble[x] = ensemble.get(x, 0) + p / len(matches)
    met_rows = []
    for x, p in sorted(ensemble.items()):
        np.savez_compressed(os.path.join(out, "val_metrica", "%s_s%d.npz" % x), p_change=p.astype(np.float32))
        met_rows.append(dict(detection(p, labels[x], scored[x], radius), match=x[0], seed=x[1]))
    summary = {"idsse": mean_rows(oof_rows), "val_metrica": mean_rows(met_rows)}
    for bench, s in summary.items():
        print("  %-11s AUC %.3f   average precision %.3f (base %.3f)   caught in top 10%% %.2f" % (
            bench, s["auc"], s["average_precision"], s["positive_rate"], s["caught_in_top10"]))
    with open(os.path.join(out, "summary.json"), "w") as handle:
        json.dump({"name": name, "datasets": os.path.basename(os.path.normpath(args.datasets)), "labels": args.labels,
                   "radius_s": args.radius_s, "groups": groups, "step": args.step, "cv_matches": matches,
                   "base_rate": base, "oof": oof_rows, "val_metrica": met_rows, "summary": summary}, handle, indent=2)
    print("saved %s  (%.0f s)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
