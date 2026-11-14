#!/usr/bin/env python3
"""Temporal decoder: per-frame probabilities -> one H / A / D label per frame (HMM + Viterbi).

    python src/mtah/decoder.py --log                                  # the candidate
    python src/mtah/decoder.py --pred external/predictions_v2_1/cv_gbm_harmonized_ball-near-context-players-tags --fit-on both --log

With no options it decodes the submission's candidate: the predictions train_classifier.py writes
with no options, and costs fitted on IDSSE against the harmonized labels. Anything else has to be
asked for. --datasets defaults to the folder the predictions were made on (their summary.json), or else to
the current datasets (tools/build_datasets.py DEFAULT_DATASETS), which must be built and up to
date; predictions and datasets from different folders are refused.

Three states, H, A and D. The emission of each frame is the classifier's log-probability; a path
pays a cost (in the same log units) every time it changes state:

    switch          H <-> A, a change of possession without a stoppage
    to_dead         H / A -> D, the ball goes dead
    from_dead       D -> H / A, play restarts
    from_dead_tag   D -> H / A inside a Smart Tagging SET PIECES window widened by tag_margin_s:
                    restarts are cheaper where the tagging says there is one

Viterbi finds the cheapest path over each half. The costs are fitted by coordinate descent on the
official spellF1 (macroF1 breaks ties) on the IDSSE predictions made with each match left out
(train_classifier.py, oof/), so val_metrica (the mean of the six rounds) scores the decoder out of
sample. --fit-on val_metrica fits on the other bench, and --fit-on both averages the two, which
then weigh the same and leave no bench out of sample; --target picks the IDSSE labels it is fitted
against. The result is always scored on both label versions. test and test_metrica are never read.

PROVISIONAL: where the tracker loses the ball in the challenge match is not known yet
(tools/ball_gap_diagnostic.py), and the synthetic SET PIECES windows sit +/-2-3 s around the
restart by construction -- the real tagging's timing is unknown. Refit once the diagnostic is in.

Decoding runs on blocks of --block frames (log-probabilities summed, so the costs mean the same
at any block size); the result is expanded back to frames.

--change DIR (src/mtah/change_detector.py output) makes the switch cost depend on the input: in each
block it is max(0, switch - change_k * s), s being the detector's log-odds that the ball changes team
there, centred on its base rate (s = 0 where the detector is at its average, so switch keeps its
meaning). change_k joins the fit (CHANGE_GRID); change_k 0 is the plain decoder, bit for bit. The
fitted file is decoder_<fit-on>[_raw]_<DIR name>.json and names DIR. Without --change nothing
changes.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for sub in ("src", "tools", "evaluation", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from mtah.features import MODEL_GROUPS  # noqa: E402
from mtah.train_classifier import (FPS, append_log, by_match, mean_of, resolve_datasets, score,  # noqa: E402
                                   set_pieces, smooth50)

STATES = np.array(["H", "A", "D"])
COSTS = ("switch", "to_dead", "from_dead", "from_dead_tag")
GRID = {"switch": (4, 8, 16, 32, 64, 128, 256), "to_dead": (2, 4, 8, 16, 32, 64, 128, 256, 512),
        "from_dead": (4, 8, 16, 32, 64, 128, 256), "from_dead_tag": (0, 1, 2, 4, 8, 16, 32),
        "tag_margin_s": (0, 1, 2, 4, 8)}
START = {"switch": 32, "to_dead": 16, "from_dead": 32, "from_dead_tag": 4, "tag_margin_s": 1}
MIN_P = 1e-4
CHANGE_GRID = (0, 0.5, 1, 2, 4, 8)      # change_k, only with --change
CHANGE_CLIP = 1e-4                      # p_change kept in [CHANGE_CLIP, 1 - CHANGE_CLIP] before the log-odds


def transitions(costs):
    """Log transition matrices (K, from, to) outside and inside a tag zone. costs: (K, 4)."""
    sw, td, fd, ft = (costs[:, i, None, None] for i in range(4))
    out = np.zeros((len(costs), 3, 3))
    out = out - sw * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) - td * np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    live = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    return out - fd * live, out - ft * live


def viterbi(logp, zone, costs, shift=None, ks=None):
    """Best state path for each of K cost settings at once. logp (n, 3), zone (n,) bool, costs
    (K, 4). With shift (n,) and ks (K,), the switch cost of step t is max(0, switch - k * shift[t]).
    Returns (K, n) state indices."""
    n, k = len(logp), len(costs)
    costs = np.asarray(costs, dtype=np.float64)
    plain, tagged = transitions(costs)
    if shift is not None:
        ks = np.asarray(ks, dtype=np.float64)
    back = np.empty((n, k, 3), dtype=np.int8)
    delta = np.repeat(logp[None, 0], k, axis=0)
    for t in range(1, n):
        trans = tagged if zone[t] else plain
        if shift is not None:
            trans = trans.copy()
            trans[:, 0, 1] = trans[:, 1, 0] = -np.maximum(costs[:, 0] - ks * shift[t], 0.0)
        cand = delta[:, :, None] + trans
        back[t] = cand.argmax(axis=1)
        delta = np.take_along_axis(cand, back[t][:, None, :].astype(np.intp), axis=1)[:, 0, :] + logp[t]
    path = np.empty((k, n), dtype=np.int8)
    path[:, -1] = delta.argmax(axis=1)
    rows = np.arange(k)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t][rows, path[:, t]]
    return path


def tag_zone(tags, n, margin_s):
    zone = np.zeros(n, dtype=bool)
    m = int(round(margin_s * FPS))
    for lo, hi, _ in tags:
        zone[max(lo - m, 0):max(min(hi + m, n), 0)] = True
    return zone


def change_shift(p_change, base_rate):
    """The detector's log-odds per frame, centred on its base rate."""
    def logit(x):
        return np.log(x) - np.log1p(-x)
    p = np.clip(np.asarray(p_change, dtype=np.float64), CHANGE_CLIP, 1 - CHANGE_CLIP)
    return logit(p) - logit(base_rate)


def decode(proba, halves, tags, costs, margin_s, block=5, change=None, ks=None):
    """Labels (K, n) for K cost settings; 'X' outside the halves. change: change_shift() per frame
    (a block takes its largest value) and ks the change_k of each setting, or None."""
    costs = np.atleast_2d(costs)
    n = len(proba)
    out = np.full((len(costs), n), "X", dtype="<U1")
    zone = tag_zone(tags, n, margin_s)
    logp = np.log(np.clip(proba, MIN_P, 1.0))
    for a, b in halves:
        m = b + 1 - a
        nb = -(-m // block)
        pad = nb * block - m
        lp = np.concatenate([logp[a:b + 1], np.zeros((pad, 3))]).reshape(nb, block, 3).sum(axis=1)
        zb = np.concatenate([zone[a:b + 1], np.zeros(pad, dtype=bool)]).reshape(nb, block).any(axis=1)
        sb = None
        if change is not None:
            sb = np.concatenate([change[a:b + 1], np.full(pad, -np.inf)]).reshape(nb, block).max(axis=1)
        path = viterbi(lp, zb, costs, sb, ks)
        out[:, a:b + 1] = STATES[np.repeat(path, block, axis=1)[:, :m]]
    return out


VERSIONS = ("harmonized", "raw")
BENCHES = ("idsse", "val_metrica")


def load_change(change_dir, datasets):
    """change_detector.py's summary, refused if it was made on other datasets."""
    with open(os.path.join(change_dir, "summary.json")) as handle:
        summary = json.load(handle)
    if summary["datasets"] != os.path.basename(os.path.normpath(datasets)):
        raise SystemExit("%s was made on external/%s, not on %s" % (change_dir, summary["datasets"], datasets))
    return summary


def load_bench(pred_dir, datasets, change_dir=None):
    """{(bench, match, seed): proba, gt labels (both versions), graded mask, halves, tags} for oof/
    and val_metrica/; with change_dir, also the detector's change_shift()."""
    import build_datasets as bd
    base = load_change(change_dir, datasets)["base_rate"] if change_dir else None
    items = {}
    for bench, pattern in (("idsse", "oof/*_s*.npz"), ("val_metrica", "val_metrica/*_s?.npz")):
        for path in sorted(glob.glob(os.path.join(pred_dir, pattern))):
            match, seed = os.path.basename(path)[:-4].rsplit("_s", 1)
            seed = int(seed)
            m = bd.load_match(datasets, match, seed)
            halves = [(a, b) for _, a, b in m["meta"]["halves"]]
            graded = np.zeros(len(m["labels"]), dtype=bool)
            for a, b in halves:
                graded[a:b + 1] = True
            gt = {"harmonized": m["labels"], "raw": bd.load_match(datasets, match, seed, labels="raw")["labels"]}
            items[bench, match, seed] = {"p": np.load(path)["p"], "gt": gt, "graded": graded, "halves": halves,
                                         "tags": set_pieces(m["smart_tagging"], m["meta"]["home"], m["meta"]["away"])}
            if change_dir:
                same = os.path.join(change_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path))
                items[bench, match, seed]["change"] = change_shift(np.load(same)["p_change"], base)
    return items


def decode_items(items, setting, block):
    """{key: labels} for one setting."""
    costs = [[setting[c] for c in COSTS]]
    ks = [setting.get("change_k", 0)]
    return {key: decode(it["p"], it["halves"], it["tags"], costs, setting["tag_margin_s"], block,
                        it.get("change"), ks)[0]
            for key, it in items.items()}


def evaluate(items, settings, block, fit_on=BENCHES, target="harmonized"):
    """Scores for each setting (dicts with the four costs and tag_margin_s), per bench. IDSSE is
    scored against the `target` labels; Metrica always against its own (harmonized) ones."""
    results = [{b: [] for b in BENCHES} for _ in settings]
    margins = sorted({s["tag_margin_s"] for s in settings})
    for (bench, match, seed), it in items.items():
        gt = it["gt"][target if bench == "idsse" else "harmonized"]
        for margin in margins:
            idx = [i for i, s in enumerate(settings) if s["tag_margin_s"] == margin]
            costs = [[settings[i][c] for c in COSTS] for i in idx]
            ks = [settings[i].get("change_k", 0) for i in idx]
            preds = decode(it["p"], it["halves"], it["tags"], costs, margin, block, it.get("change"), ks)
            for i, pred in zip(idx, preds):
                results[i][bench].append(dict(score(gt, pred, it["graded"]), match=match, seed=seed))
    out = []
    for rows in results:
        r = {"idsse": by_match(rows["idsse"], ("spell_f1", "macro_f1", "possessions")),
             "val_metrica": mean_of(rows["val_metrica"], ("spell_f1", "macro_f1", "possessions")),
             "rows": rows}
        r["objective"] = float(np.mean([r[b]["spell_f1"] for b in fit_on]))
        r["tie_break"] = float(np.mean([r[b]["macro_f1"] for b in fit_on]))
        out.append(r)
    return out


def fit(items, block, fit_on, target, passes=3):
    """Coordinate descent over GRID from START (plus change_k when the items carry a change signal);
    returns the best setting and the search trace."""
    grid, best = dict(GRID), dict(START)
    if any("change" in it for it in items.values()):
        grid["change_k"], best["change_k"] = CHANGE_GRID, 0
    best_score = evaluate(items, [best], block, fit_on, target)[0]
    trace = []
    for p in range(passes):
        changed = False
        for name, values in grid.items():
            settings = [dict(best, **{name: v}) for v in values]
            scores = evaluate(items, settings, block, fit_on, target)
            for s, r in zip(settings, scores):
                trace.append({"pass": p + 1, "param": name, "value": s[name], "objective": r["objective"],
                              "idsse": r["idsse"]["spell_f1"], "val_metrica": r["val_metrica"]["spell_f1"]})
            i = max(range(len(values)), key=lambda j: (scores[j]["objective"], scores[j]["tie_break"]))
            print("  pass %d  %-14s %s -> %g   objective %.4f   (idsse %.4f   val_metrica %.4f)" % (
                p + 1, name, " ".join("%g:%.3f" % (v, r["objective"]) for v, r in zip(values, scores)), values[i],
                scores[i]["objective"], scores[i]["idsse"]["spell_f1"], scores[i]["val_metrica"]["spell_f1"]))
            if (scores[i]["objective"], scores[i]["tie_break"]) > (best_score["objective"], best_score["tie_break"]):
                best, best_score, changed = settings[i], scores[i], True
        if not changed:
            break
    return best, trace


def decoder_file(pred_dir, fit_on, target, change=None):
    suffix = "_" + os.path.basename(os.path.normpath(change)) if change else ""
    return os.path.join(pred_dir, "decoder_%s%s%s.json" % (fit_on, "_raw" if target == "raw" else "", suffix))


def candidate_predictions():
    """Where train_classifier.py writes the candidate when it is run with no options."""
    import build_datasets as bd
    folder = os.path.basename(bd.DEFAULT_DATASETS).replace("datasets", "predictions", 1)
    return os.path.join(ROOT, "external", folder, "cv_gbm_raw_" + "-".join(MODEL_GROUPS))


def pred_datasets(pred_dir, datasets=None):
    """The classifier's summary.json and the datasets folder its predictions go with: the one they
    were made on, unless another is given (refused if it differs from the one they were made on)."""
    with open(os.path.join(pred_dir, "summary.json")) as handle:
        summary = json.load(handle)
    made_on = summary.get("datasets")
    if made_on and datasets and os.path.basename(os.path.normpath(datasets)) != made_on:
        raise SystemExit("%s was made on external/%s, not on %s" % (pred_dir, made_on, datasets))
    if made_on and datasets is None:
        import build_datasets as bd
        if made_on != os.path.basename(bd.DEFAULT_DATASETS):
            datasets = os.path.join(ROOT, "external", made_on)
    return summary, resolve_datasets(datasets)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", default=None, help="a train_classifier.py output folder (oof/ and val_metrica/); "
                                                     "default: the candidate's")
    parser.add_argument("--datasets", default=None,
                        help="default: the folder the predictions were made on, else the current datasets (v2.1)")
    parser.add_argument("--block", type=int, default=5, help="frames per decoding step")
    parser.add_argument("--fit-on", choices=("both", "idsse", "val_metrica"), default="idsse",
                        help="the bench(es) the costs are fitted on (default: idsse, the candidate)")
    parser.add_argument("--target", choices=VERSIONS, default="harmonized",
                        help="the IDSSE labels the costs are fitted against (Metrica: always its own)")
    parser.add_argument("--change", default=None,
                        help="a change_detector.py output folder: the switch cost then depends on it (change_k)")
    parser.add_argument("--log", action="store_true", help="append the results to experiments/log.csv")
    args = parser.parse_args()
    fit_on = BENCHES if args.fit_on == "both" else (args.fit_on,)
    args.pred = args.pred or candidate_predictions()
    if not os.path.isfile(os.path.join(args.pred, "summary.json")):
        raise SystemExit("no predictions in %s: run python src/mtah/train_classifier.py first" % args.pred)

    summary, args.datasets = pred_datasets(args.pred, args.datasets)
    trained_on = summary["labels"]
    print("datasets %s" % args.datasets)
    name = os.path.basename(os.path.normpath(args.pred))
    t0 = time.time()
    items = load_bench(args.pred, args.datasets, args.change)
    print("%s: %d sequences, classifier on %s labels, block %d frames, fitted on %s (IDSSE vs %s labels)%s" % (
        name, len(items), trained_on, args.block, " + ".join(fit_on), args.target,
        ", switch cost from %s" % os.path.basename(os.path.normpath(args.change)) if args.change else ""))
    best, trace = fit(items, args.block, fit_on, args.target)
    print("best: %s" % ", ".join("%s %g" % kv for kv in best.items()))

    cache = os.path.dirname(os.path.normpath(args.pred))
    report, log = {}, []
    change = "%s + HMM decoder%s fitted on %s vs %s labels (provisional: %s)" % (
        name, " with the switch cost from %s" % os.path.basename(os.path.normpath(args.change)) if args.change else "",
        " + ".join(fit_on), args.target, ", ".join("%s %g" % kv for kv in best.items()))
    for version in VERSIONS:
        scores = evaluate(items, [best], args.block, BENCHES, version)[0]
        report[version] = {"scores": {b: scores[b] for b in BENCHES}, "rows": scores["rows"]}
        for b in BENCHES:
            keys = [k for k in items if k[0] == b]
            base = [dict(smooth50(args.datasets, m, s, cache)[version if b == "idsse" else "harmonized"], match=m)
                    for _, m, s in keys]
            base = (by_match if b == "idsse" else mean_of)(base, ("spell_f1", "macro_f1", "possessions"))
            gt = [items[k]["gt"][version if b == "idsse" else "harmonized"] for k in keys]
            truth = (by_match if b == "idsse" else mean_of)(
                [dict(possessions=int(np.sum((g[1:] != g[:-1]) & np.isin(g[1:], ("H", "A")))), match=k[1])
                 for g, k in zip(gt, keys)], ("possessions",))
            report[version][b] = {"smooth50": base, "truth_possessions": truth["possessions"]}
            s = scores[b]
            print("  %-10s %-12s spellF1 %.3f   macroF1 %.3f   possessions %4.0f (truth %4.0f)   | smooth50 spellF1 %.3f"
                  "   macroF1 %.3f" % (version, b, s["spell_f1"], s["macro_f1"], s["possessions"], truth["possessions"],
                                       base["spell_f1"], base["macro_f1"]))
            if b == "idsse" or version == "harmonized":
                log.append((change, "%s/%s labels" % ("cv_idsse oof" if b == "idsse" else "val_metrica ensemble of rounds",
                                                      version if b == "idsse" else "metrica"), s, base))
    out = decoder_file(args.pred, args.fit_on, args.target, args.change)
    saved = {"provisional": True, "block": args.block, "trained_on": trained_on, "fit_on": fit_on,
             "target": args.target, "params": best, "report": report, "trace": trace}
    if args.change:
        saved["change"] = os.path.relpath(os.path.abspath(args.change), ROOT).replace(os.sep, "/")
        saved["change_base_rate"] = load_change(args.change, args.datasets)["base_rate"]   # the zero of change_k
    with open(out, "w") as handle:
        json.dump(saved, handle, indent=2)
    print("saved %s  (%.0f s)" % (out, time.time() - t0))
    if args.log:
        append_log(log)


if __name__ == "__main__":
    main()
