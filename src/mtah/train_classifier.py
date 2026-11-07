#!/usr/bin/env python3
"""Per-frame classifier, leave-one-match-out over the six IDSSE train + val matches.

    python src/mtah/train_classifier.py --datasets external/datasets --log
    python src/mtah/train_classifier.py --labels raw --without gap --log

A gradient-boosted tree model on src/mtah/features.py, predicting H / A / D per frame. Each round
leaves one IDSSE match out (all its seeds together), trains on the other five and predicts the
held-out match and val_metrica. Saved under --out/<name>/:

    oof/<match>_s<seed>.npz                  the held-out match, from the round that left it out
    val_metrica/<game>_s<seed>_r<match>.npz  each round's prediction for val_metrica
    val_metrica/<game>_s<seed>.npz           the mean of the six rounds (what the decoder uses)
    summary.json                             per-match and per-round scores

These are the decoder's inputs (src/mtah/decoder.py). The scores printed here are those of the raw
argmax, with no temporal decoding, so spellF1 is expected to be low; every row also carries the
official baseline (smooth50) on the same frames and labels, as a measure of how hard they are.
test and test_metrica are never read.

It reads only external/ (public datasets). The hyperparameters default to the library's, plus a
fixed seed; --tag names a run with other values. Nothing here is tuned on the challenge match.
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for sub in ("src", "tools", "evaluation", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from mtah.features import COLUMNS, GROUPS, MODEL_GROUPS, frame_features  # noqa: E402

CLASSES = np.array(["H", "A", "D"])
FPS = 25
CV_SPLITS = ("train", "val")
LOG_HEADER = ["date", "author", "commit", "change", "bench", "spell_f1", "macro_f1", "possessions",
              "smooth50_spell_f1", "smooth50_macro_f1"]


def set_pieces(xml_path, home, away):
    out = []
    for inst in ET.parse(xml_path).iter("instance"):
        if (inst.findtext("code") or "").strip().lower() != "set pieces":
            continue
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        team = {home: "H", away: "A"}.get((labels.get("Team") or "").strip())
        out.append((int(round(float(inst.findtext("start")) * FPS)), int(round(float(inst.findtext("end")) * FPS)), team))
    return out


def load(datasets, match, seed, groups=GROUPS):
    """Features, both label versions and the graded mask for one match and seed."""
    import build_datasets as bd
    m = bd.load_match(datasets, match, seed)
    meta = m["meta"]
    halves = [(a, b) for _, a, b in meta["halves"]]
    feats = frame_features(m["players"], m["slot_team"], m["ball"], tuple(meta["pitch_m"]), halves,
                           set_pieces(m["smart_tagging"], meta["home"], meta["away"]), groups)
    raw = bd.load_match(datasets, match, seed, labels="raw")["labels"]
    graded = np.zeros(len(m["labels"]), dtype=bool)
    for a, b in halves:
        graded[a:b + 1] = True
    return {"features": feats, "labels": {"harmonized": m["labels"], "raw": raw}, "graded": graded,
            "source": meta["source"]}


def score(gt_labels, pred, graded):
    """Official scores of a label sequence (pred: labels, 'X' outside the graded frames)."""
    import possession_metrics as pm
    r = pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred))
    return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"], "accuracy": r["accuracy"],
            "possessions": int(np.sum((pred[1:] != pred[:-1]) & np.isin(pred[1:], ("H", "A"))))}


def score_proba(gt_labels, proba, graded):
    from sklearn.metrics import log_loss
    s = score(gt_labels, np.where(graded, CLASSES[np.argmax(proba, axis=1)], "X"), graded)
    ok = graded & np.isin(gt_labels, CLASSES)
    y = (gt_labels[ok] == "A") * 1 + (gt_labels[ok] == "D") * 2                 # index in CLASSES
    s["log_loss"] = log_loss(y, proba[ok], labels=[0, 1, 2])
    return s


def smooth50(datasets, match, seed, cache_dir):
    """The official baseline (nearest player, +/-50 frames) scored on both label versions; cached,
    since it depends on nothing but the dataset -- keyed by the dataset folder too (datasets, datasets_v2)."""
    path = os.path.join(cache_dir, "smooth50.json")
    cache = {}
    if os.path.isfile(path):
        with open(path) as handle:
            cache = json.load(handle)
    key = "%s/%s_s%d" % (os.path.basename(os.path.normpath(datasets)), match, seed)
    if key not in cache:
        import build_datasets as bd
        from baseline import smooth
        m = bd.load_match(datasets, match, seed)
        pred = smooth(bd.nearest_player_codes(m), 50)
        graded = np.zeros(len(pred), dtype=bool)
        for _, a, b in m["meta"]["halves"]:
            graded[a:b + 1] = True
        raw = bd.load_match(datasets, match, seed, labels="raw")["labels"]
        cache[key] = {"harmonized": score(m["labels"], pred, graded), "raw": score(raw, pred, graded)}
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(cache, handle, indent=1, sort_keys=True)
    return cache[key]


def git_commit():
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=ROOT, text=True).strip()
    except Exception:                                                  # noqa: BLE001
        return "?"


def append_log(rows):
    """rows: [(change, bench, scores, smooth50 scores)] appended to experiments/log.csv."""
    path = os.path.join(ROOT, "experiments", "log.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.isfile(path)
    author = subprocess.run(["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(path, "a", newline="") as handle:
        w = csv.writer(handle)
        if new:
            w.writerow(LOG_HEADER)
        for change, bench, s, b in rows:
            w.writerow([datetime.date.today().isoformat(), author, git_commit(), change, bench,
                        "%.4f" % s["spell_f1"], "%.4f" % s["macro_f1"], "%.0f" % s["possessions"],
                        "%.4f" % b["spell_f1"], "%.4f" % b["macro_f1"]])


def mean_of(rows, keys=("spell_f1", "macro_f1", "accuracy", "log_loss", "possessions")):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys if all(k in r for r in rows)}


def by_match(rows, keys=("spell_f1", "macro_f1", "accuracy", "log_loss", "possessions")):
    """Mean over seeds within a match, then over matches: every match weighs the same."""
    matches = sorted({r["match"] for r in rows})
    return mean_of([mean_of([r for r in rows if r["match"] == m], keys) for m in matches], keys)


def run(name, labels, groups, data, splits, args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import pandas as pd

    pool = [tuple(x) for split in CV_SPLITS for x in splits[split]]
    matches = sorted({m for m, _ in pool})
    metrica = [tuple(x) for x in splits.get("val_metrica", [])]
    cols = [c for g in groups for c in COLUMNS[g]]
    out = os.path.join(args.out, name)
    for sub in ("oof", "val_metrica"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    oof_rows, metrica_rows, ensemble = [], [], {}
    t0 = time.time()
    for held in matches:
        xs, ys = [], []
        for match in matches:
            if match == held:
                continue
            seeds = [s for m, s in pool if m == match]
            for seed in seeds:                       # every match weighs the same, whatever its seeds
                d = data[match, seed]
                keep = np.flatnonzero(d["graded"] & np.isin(d["labels"][labels], CLASSES))[::args.step * len(seeds)]
                xs.append(d["features"].iloc[keep][cols])
                ys.append(d["labels"][labels][keep])
        model = HistGradientBoostingClassifier(random_state=0, early_stopping=False, learning_rate=args.learning_rate,
                                               max_iter=args.max_iter, max_leaf_nodes=args.max_leaf_nodes,
                                               min_samples_leaf=args.min_samples_leaf)
        model.fit(pd.concat(xs, ignore_index=True), np.concatenate(ys))
        order = [list(model.classes_).index(c) for c in CLASSES]
        for (match, seed) in [x for x in pool if x[0] == held] + metrica:
            d = data[match, seed]
            proba = model.predict_proba(d["features"][cols])[:, order].astype(np.float32)
            s = dict(score_proba(d["labels"][labels], proba, d["graded"]), match=match, seed=seed, held_out=held)
            if match == held:
                np.savez_compressed(os.path.join(out, "oof", "%s_s%d.npz" % (match, seed)), p=proba, classes=CLASSES)
                oof_rows.append(s)
            else:
                np.savez_compressed(os.path.join(out, "val_metrica", "%s_s%d_r%s.npz" % (match, seed, held)),
                                    p=proba, classes=CLASSES)
                ensemble[match, seed] = ensemble.get((match, seed), 0) + proba / len(matches)
                metrica_rows.append(s)
            print("  round %-7s %-11s %-8s s%d   spellF1 %.3f   macroF1 %.3f   logloss %.3f   possessions %d" % (
                held, "held out" if match == held else "val_metrica", match, seed, s["spell_f1"], s["macro_f1"],
                s["log_loss"], s["possessions"]))
    ens_rows = []
    for (match, seed), proba in ensemble.items():
        np.savez_compressed(os.path.join(out, "val_metrica", "%s_s%d.npz" % (match, seed)), p=proba, classes=CLASSES)
        d = data[match, seed]
        ens_rows.append(dict(score_proba(d["labels"][labels], proba, d["graded"]), match=match, seed=seed))

    base = {x: smooth50(args.datasets, x[0], x[1], args.out)[labels] for x in pool + metrica}
    oof_base = by_match([dict(base[x], match=x[0]) for x in pool], ("spell_f1", "macro_f1", "possessions"))
    met_base = mean_of([base[x] for x in metrica], ("spell_f1", "macro_f1", "possessions"))
    rounds = [mean_of([r for r in metrica_rows if r["held_out"] == h]) for h in matches]
    summary = {"oof": by_match(oof_rows), "val_metrica_rounds": mean_of(rounds),
               "val_metrica_rounds_sd": {k: float(np.std([r[k] for r in rounds])) for k in rounds[0]},
               "val_metrica_ensemble": mean_of(ens_rows), "smooth50": {"oof": oof_base, "val_metrica": met_base}}
    for key, s, b in (("oof", summary["oof"], oof_base), ("val_metrica_rounds", summary["val_metrica_rounds"], met_base),
                      ("val_metrica_ensemble", summary["val_metrica_ensemble"], met_base)):
        print("  %-21s spellF1 %.3f   macroF1 %.3f   logloss %.3f   possessions %5.0f   | smooth50 spellF1 %.3f   "
              "macroF1 %.3f   possessions %4.0f" % (key, s["spell_f1"], s["macro_f1"], s["log_loss"], s["possessions"],
                                                     b["spell_f1"], b["macro_f1"], b["possessions"]))
    print("  val_metrica spread over rounds (sd): spellF1 %.4f   macroF1 %.4f" % (
        summary["val_metrica_rounds_sd"]["spell_f1"], summary["val_metrica_rounds_sd"]["macro_f1"]))
    with open(os.path.join(out, "summary.json"), "w") as handle:
        json.dump({"name": name, "labels": labels, "groups": groups, "step": args.step, "cv_matches": matches,
                   "model": {k: getattr(args, k) for k in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf")},
                   "oof": oof_rows, "val_metrica": metrica_rows, "val_metrica_ensemble": ens_rows, "summary": summary},
                  handle, indent=2)
    print("  predictions in %s  (%.0f s)" % (out, time.time() - t0))
    change = "%s, leave-one-match-out x%d (argmax, no decoder)" % (name, len(matches))
    return [(change, "cv_idsse oof/%s labels" % labels, summary["oof"], oof_base),
            (change, "val_metrica mean of rounds/%s labels" % labels, summary["val_metrica_rounds"], met_base)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", default=os.path.join(ROOT, "external", "datasets"))
    parser.add_argument("--labels", nargs="+", choices=("harmonized", "raw"), default=["harmonized", "raw"])
    parser.add_argument("--groups", nargs="+", default=list(MODEL_GROUPS), choices=GROUPS,
                        help="feature groups of the full model (default: every group but touches)")
    parser.add_argument("--without", nargs="*", default=["gap"], choices=GROUPS,
                        help="besides the model with every group, one model without each of these")
    parser.add_argument("--step", type=int, default=5, help="train on one frame in STEP (per match)")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--tag", default="", help="added to the run's name, for runs with other hyperparameters")
    parser.add_argument("--out", default=os.path.join(ROOT, "external", "predictions"))
    parser.add_argument("--log", action="store_true", help="append the results to experiments/log.csv")
    args = parser.parse_args()

    with open(os.path.join(args.datasets, "splits.json")) as handle:
        splits = json.load(handle)
    t0 = time.time()
    items = [tuple(x) for split in CV_SPLITS + ("val_metrica",) for x in splits.get(split, [])]
    groups = tuple(g for g in GROUPS if g in args.groups)          # always in GROUPS order
    data = {x: load(args.datasets, *x, groups=groups) for x in items}
    variants = [groups] + [tuple(g for g in groups if g != w) for w in args.without if w in groups]
    print("%d matches x seeds loaded (%.0f s)" % (len(items), time.time() - t0))
    log = []
    for labels in args.labels:
        for groups in variants:
            name = "cv_gbm_%s_%s%s" % (labels, "-".join(groups), "_" + args.tag if args.tag else "")
            print("\n" + name)
            log += run(name, labels, groups, data, splits, args)
    if args.log:
        append_log(log)


if __name__ == "__main__":
    main()
