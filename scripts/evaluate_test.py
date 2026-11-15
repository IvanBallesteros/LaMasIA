#!/usr/bin/env python3
"""The test matches, scored once with the submitted pipeline.

    python scripts/make_submission.py --train               # the final models, first
    python scripts/evaluate_test.py                         # J03WR9 (IDSSE) and METRICA3, once
    python scripts/evaluate_test.py --splits val            # a dry run of this code on the val matches

The submitted pipeline (docs/DECISIONS.md, 2026-11-15) is the classifier and the turnover detector of
scripts/make_submission.py --train, trained on the six IDSSE train + val matches only, with
the decoder costs of scripts/decoder_idsse_change.json. Each match goes through the submission's own
functions (make_submission.predict_proba, predict_change and decode_labels) on the features of the
validation path, which tests/test_make_submission.py shows give the submission's labels exactly.

Scores come from the official scorer (evaluation/possession_metrics.py): spellF1 with its bootstrap
interval over possessions (1000 draws, the scorer's 5th to 95th percentiles), macroF1 and the number
of possessions, with the official baseline smooth50 on the same frames as a reference. J03WR9 is
scored on both IDSSE label versions, METRICA3 on its own labels.

Once. The test matches are the only ones nothing was chosen on, and they stay that way only if
nothing changes after they are seen. The result is saved in external/test_evaluation/; when it is
already there, the script prints it and stops. --again scores again, and is only for a clear bug in
this script, never for another model. --log appends the rows to experiments/log.csv.

The dry run (--splits val) scores matches the final models were trained on: its numbers mean nothing
and are only printed, never saved or logged. Public data only (external/); nothing from the
challenge match.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_submission as ms  # noqa: E402  -- first: it sets OMP_NUM_THREADS=1 before numpy is imported

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import build_datasets as bd  # noqa: E402
import possession_metrics as pm  # noqa: E402
from mtah import train_classifier as tc  # noqa: E402

ROOT = ms.ROOT
OUT = os.path.join(ROOT, "external", "test_evaluation")
TEST_SPLITS = ("test", "test_metrica")
DRAWS = 1000


def possessions(labels):
    """Possessions as the rest of the code counts them: every start of an H or A run."""
    return int(np.sum((labels[1:] != labels[:-1]) & np.isin(labels[1:], ("H", "A"))))


def score(gt, pred, graded, smooth50):
    """Official scores of one prediction against one label version, with smooth50 alongside."""
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=DRAWS)
    return {"spell_f1": r["spell_f1_mean"], "spell_f1_low": r["spell_bootstrap_low"],
            "spell_f1_high": r["spell_bootstrap_high"], "macro_f1": r["macro_f1"],
            "possessions": possessions(np.where(graded, pred, "X")), "truth_possessions": possessions(np.where(graded, gt, "X")),
            "smooth50_spell_f1": smooth50["spell_f1"], "smooth50_macro_f1": smooth50["macro_f1"]}


def sequences(splits):
    with open(os.path.join(ms.DATASETS, "splits.json")) as handle:
        table = json.load(handle)
    return [(split, match, seed) for split in splits for match, seed in table[split]]


def run(items, final, cache):
    """{"<split> <match> s<seed> / <labels>": scores} for every sequence."""
    model, detector, costs = final
    rows = {}
    for split, match, seed in items:
        t0 = time.time()
        m = bd.load_match(ms.DATASETS, match, seed)
        meta = m["meta"]
        halves = [(a, b) for _, a, b in meta["halves"]]
        tags = tc.set_pieces(m["smart_tagging"], meta["home"], meta["away"])
        d = tc.load(ms.DATASETS, match, seed, groups=ms.MODEL_GROUPS)
        proba = ms.predict_proba(model, d["features"])
        p_change = ms.predict_change(detector, tc.load(ms.DATASETS, match, seed, groups=ms.CHANGE_GROUPS)["features"])
        pred = ms.decode_labels(proba, halves, tags, costs, p_change)
        base = tc.smooth50(ms.DATASETS, match, seed, cache)
        versions = ("harmonized",) if meta["source"] == "metrica" else ("harmonized", "raw")
        for version in versions:                    # Metrica's own labels are its "harmonized" ones
            name = "metrica" if meta["source"] == "metrica" else version
            rows["%s %s s%d / %s" % (split, match, seed, name)] = score(d["labels"][version], pred, d["graded"], base[version])
        print("  %s %s s%d scored (%.0f s)" % (split, match, seed, time.time() - t0))
    return rows


def show(result):
    print("\nfinal pipeline: classifier %s, detector %s, decoder %s" % (
        result["classifier_sha256"][:16], result["detector_sha256"][:16], result["decoder_params"]))
    print("commit %s, datasets %s, %s\n" % (result["commit"], result["datasets"], result["date"]))
    print("%-36s %8s  %-17s %8s %12s   | %s" % ("match / labels", "spellF1", "90% interval", "macroF1", "poss / truth",
                                              "smooth50 spellF1 / macroF1"))
    for key, s in result["rows"].items():
        print("%-36s %8.3f  [%.3f, %.3f]  %8.3f %5d / %-5d  | %.3f / %.3f" % (
            key, s["spell_f1"], s["spell_f1_low"], s["spell_f1_high"], s["macro_f1"], s["possessions"],
            s["truth_possessions"], s["smooth50_spell_f1"], s["smooth50_macro_f1"]))


def log(result):
    change = ("FINAL submitted pipeline, scored once on the test matches (scripts/evaluate_test.py): "
              "make_submission --change, classifier %s, detector %s, decoder %s" % (
                  result["classifier_sha256"][:8], result["detector_sha256"][:8],
                  ", ".join("%s %g" % kv for kv in result["decoder_params"].items())))
    rows = [(change, "%s labels" % key, s, {"spell_f1": s["smooth50_spell_f1"], "macro_f1": s["smooth50_macro_f1"]})
            for key, s in result["rows"].items()]
    tc.append_log(rows)
    print("\nappended %d rows to experiments/log.csv" % len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS),
                        help="default: test and test_metrica; 'val' is a dry run of this code")
    parser.add_argument("--models-dir", default=ms.MODELS)
    parser.add_argument("--again", action="store_true", help="score again although a result exists: a clear bug only")
    parser.add_argument("--log", action="store_true", help="append the rows to experiments/log.csv")
    args = parser.parse_args()
    testing = any(s in TEST_SPLITS for s in args.splits)
    if testing and sorted(args.splits) != sorted(TEST_SPLITS):
        raise SystemExit("the test matches are scored together, once: run without --splits")
    path = os.path.join(OUT, "final_change.json")
    if testing and os.path.isfile(path) and not args.again:
        with open(path) as handle:
            result = json.load(handle)
        print("already scored on %s, so this prints the saved result (%s).\n"
              "Scoring again is only for a clear bug in this script: --again." % (result["date"], path))
        show(result)
        if args.log:
            log(result)
        return

    model, manifest = ms.load_model(args.models_dir)
    detector, change_manifest = ms.load_change_model(args.models_dir)
    costs = ms.load_change_costs()
    datasets = os.path.basename(os.path.normpath(ms.DATASETS))
    for man in (manifest, change_manifest):
        if man["datasets"] != datasets:
            raise SystemExit("%s was trained on external/%s, not on %s: run make_submission.py --train --change" % (
                man["name"], man["datasets"], datasets))
    items = sequences(args.splits)
    trained = {m for man in (manifest, change_manifest) for m, _ in man["sequences"]}
    seen = sorted({match for _, match, _ in items} & trained)
    if testing and seen:
        raise SystemExit("the final models were trained on %s: these are not test matches" % ", ".join(seen))
    if not testing:
        print("DRY RUN on %s: the final models were trained on %s, so these numbers mean nothing.\n" % (
            " ".join(args.splits), ", ".join(seen) or "none of them"))

    t0 = time.time()
    rows = run(items, (model, detector, costs), OUT if testing else os.path.join(OUT, "dry_run"))
    result = {"date": datetime.date.today().isoformat(), "commit": tc.git_commit(), "datasets": datasets,
              "splits": args.splits, "classifier": manifest["name"], "classifier_sha256": manifest["model_sha256"],
              "detector": change_manifest["name"], "detector_sha256": change_manifest["model_sha256"],
              "decoder_params": costs["params"], "bootstrap_draws": DRAWS,
              "bootstrap_percentiles": list(pm.BOOTSTRAP_PERCENTILES), "rows": rows}
    if testing:                                     # saved before it is shown: a second run only prints it
        os.makedirs(OUT, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
    show(result)
    print("\n(%.0f s)%s" % (time.time() - t0, "  saved %s" % path if testing else ""))
    if testing and args.log:
        log(result)


if __name__ == "__main__":
    main()
