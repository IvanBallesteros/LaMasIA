#!/usr/bin/env python3
"""Two decoded pipelines, match by match, with a paired bootstrap of the spellF1 difference.

    python src/mtah/compare.py \
        --a external/predictions/cv_gbm_harmonized_ball-near-context-players-gap-tags/decoder_both.json \
        --b external/predictions/cv_gbm_raw_ball-near-context-players-gap-tags/decoder_val_metrica.json

Each side is a decoder file (src/mtah/decoder.py); its classifier's predictions are decoded again
with the saved parameters. For every match the ground-truth possessions are resampled with
replacement -- the same draws for both pipelines, as the official bootstrap_spell_f1 does for one --
and the difference B - A is recomputed. A bench's difference is the mean over its matches (IDSSE:
seeds of a match averaged first), drawn match by match. Scored on both label versions.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for sub in ("src", "tools", "evaluation", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from mtah.decoder import BENCHES, VERSIONS, decode_items, load_bench  # noqa: E402

DRAWS = 1000


def spell_flags(gt_codes, pred_codes):
    """(matched flags per IoU threshold [k, n_gt], n_pred) as the official scorer computes them."""
    import possession_metrics as pm
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def spell_f1(flags, n_pred, picks):
    """spellF1 (mean over IoU thresholds) of a resample `picks` of the ground-truth possessions."""
    n_gt = len(picks)
    tp = flags[:, picks].sum(axis=1)
    fp = np.maximum(0, n_pred - tp)
    fn = n_gt - tp
    return float(np.mean(np.where(tp > 0, 2.0 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)))


def paired(gt, pred_a, pred_b, rng, draws=DRAWS):
    """Point spellF1 of A and B and the bootstrap draws of B - A on one match."""
    import possession_metrics as pm
    g = pm.label_codes(gt)
    fa, na = spell_flags(g, pm.label_codes(pred_a))
    fb, nb = spell_flags(g, pm.label_codes(pred_b))
    n = fa.shape[1]
    every = np.arange(n)
    diffs = np.empty(draws)
    for k in range(draws):
        picks = rng.randint(0, n, n)
        diffs[k] = spell_f1(fb, nb, picks) - spell_f1(fa, na, picks)
    return spell_f1(fa, na, every), spell_f1(fb, nb, every), diffs


def side(path):
    with open(path) as handle:
        d = json.load(handle)
    return os.path.dirname(os.path.abspath(path)), d["params"], d.get("block", 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, help="decoder file of pipeline A")
    parser.add_argument("--b", required=True, help="decoder file of pipeline B")
    parser.add_argument("--datasets", default=os.path.join(ROOT, "external", "datasets"))
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--out", default=None, help="write the comparison here (json)")
    args = parser.parse_args()
    import possession_metrics as pm

    (dir_a, prm_a, block_a), (dir_b, prm_b, block_b) = side(args.a), side(args.b)
    items_a, items_b = load_bench(dir_a, args.datasets), load_bench(dir_b, args.datasets)
    assert sorted(items_a) == sorted(items_b), "the two pipelines were not run on the same matches"
    pred_a, pred_b = decode_items(items_a, prm_a, block_a), decode_items(items_b, prm_b, block_b)
    print("A = %s\nB = %s\npaired bootstrap of spellF1 B - A, %d draws per match, 95%% interval" % (
        os.path.relpath(args.a, ROOT), os.path.relpath(args.b, ROOT), args.draws))
    result = {"a": args.a, "b": args.b, "draws": args.draws, "versions": {}}
    for version in VERSIONS:
        rows = []
        for key in sorted(items_a):
            bench, match, seed = key
            gt = items_a[key]["gt"][version if bench == "idsse" else "harmonized"]
            rng = np.random.RandomState(pm.BOOTSTRAP_SEED)
            a, b, diffs = paired(gt, pred_a[key], pred_b[key], rng, args.draws)
            rows.append({"bench": bench, "match": match, "seed": seed, "a": a, "b": b, "diffs": diffs})
        print("\n  labels: %s (Metrica: its own)" % version)
        print("  %-12s %-9s %8s %8s %8s   %-18s" % ("bench", "match", "A", "B", "B - A", "95% interval"))
        out = []
        for bench in BENCHES:
            sel = [r for r in rows if r["bench"] == bench]
            matches = sorted({r["match"] for r in sel})
            per_match = []
            for m in matches:
                ms = [r for r in sel if r["match"] == m]
                a = np.mean([r["a"] for r in ms])
                b = np.mean([r["b"] for r in ms])
                d = np.mean([r["diffs"] for r in ms], axis=0)
                lo, hi = np.percentile(d, (2.5, 97.5))
                per_match.append({"match": m, "a": a, "b": b, "diff": b - a, "low": float(lo), "high": float(hi), "draws": d})
                print("  %-12s %-9s %8.3f %8.3f %+8.3f   [%+.3f, %+.3f]%s" % (
                    bench, m, a, b, b - a, lo, hi, "  *" if lo > 0 or hi < 0 else ""))
            d = np.mean([x["draws"] for x in per_match], axis=0)
            lo, hi = np.percentile(d, (2.5, 97.5))
            a = np.mean([x["a"] for x in per_match])
            b = np.mean([x["b"] for x in per_match])
            print("  %-12s %-9s %8.3f %8.3f %+8.3f   [%+.3f, %+.3f]%s" % (
                bench, "mean", a, b, b - a, lo, hi, "  *" if lo > 0 or hi < 0 else ""))
            out.append({"bench": bench, "mean": {"a": a, "b": b, "diff": b - a, "low": float(lo), "high": float(hi)},
                        "matches": [{k: v for k, v in x.items() if k != "draws"} for x in per_match]})
        result["versions"][version] = out
    print("\n  * the interval excludes 0")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2, default=float)


if __name__ == "__main__":
    main()
