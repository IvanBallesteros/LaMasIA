#!/usr/bin/env python3
"""Where the decoder loses possessions: an error analysis of decoder.py on the validation benches.

    python src/mtah/decoder_analysis.py                    # the candidate, after decoder.py
    python src/mtah/decoder_analysis.py --pred external/predictions_v2_1/<run> --fit-on both

Reads the classifier's out-of-fold predictions (oof/ and val_metrica/, as decoder.py does), decodes
them with the costs decoder.py fitted (decoder_<fit-on>[_raw].json in the same folder; by default the
candidate's, fitted on IDSSE, so val_metrica is out of sample) and matches
true and decoded possessions with the official scorer's own functions
(evaluation/possession_metrics.py), so every count here adds up to the official spellF1. IDSSE is
analysed against both label versions, val_metrica against its own labels.

It answers four questions:

1. Which true possessions are missed: by duration, by how they start (a restart after a stoppage,
   or a turnover in play) and by how much of them the ball is visible.
2. How they are missed, at the headline IoU 0.5:
     absent   no decoded possession of that team overlaps it
     merged   the decoded possession that overlaps it most also overlaps another true possession of
              the same team: the stoppage or the rival spell in between was not decoded
     split    two or more decoded possessions of that team share it, none well enough
     shifted  one decoded possession, but its edges are too far off
   and what the spurious decoded possessions are (invented, merged, a fragment, shifted).
3. Who is to blame: the classifier's evidence for the true labels over the decoded ones,
   sum(log p[true] - log p[decoded]) over the possession and EVIDENCE_S seconds either side (where
   a merge's missing stoppage or rival spell lies), and over the possession's own frames alone. At
   or below zero the classifier itself prefers what the decoder said; above zero the classifier saw
   it and the transition costs overruled it.
4. How much each kind of fix could gain at most. These oracles use the true labels, so they are
   ceilings to rank ideas by, never candidates:
     snap edges  every decoded transition moved to the nearest true one into the same label within
                 SNAP_S: the ceiling of boundary refinement
     turnovers   the switch cost dropped to 0 within TURNOVER_S of every true change of team: a
                 perfect change-point detector feeding an input-dependent switch cost
     dead ball   the true alive / dead state given to the decoder (p_D 1 or 0): a perfect dead-ball
                 detector
     both        turnovers + dead ball

--change analyses the decoder fitted with that change detector (decoder.py --change).

Only external/ is read (public benches); test and test_metrica are never read. Prints aggregates
and writes, next to the decoder file, <decoder file>_analysis.json and <decoder file>_analysis_spells.csv
(one row per true or decoded possession), both outside git.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for sub in ("src", "tools", "evaluation", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import possession_metrics as pm  # noqa: E402
from mtah.decoder import (COSTS, MIN_P, STATES, VERSIONS, candidate_predictions, decode, decoder_file,  # noqa: E402
                          load_bench, pred_datasets, tag_zone)
from mtah.train_classifier import FPS, by_match, mean_of, score  # noqa: E402

NAMES = np.array(pm.LABELS)                      # code -> label; H, A, D share their index with STATES
DURATIONS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf))   # seconds
BALL = ((0, 0.25), (0.25, 0.75), (0.75, 1.01))   # fraction of the possession with the ball visible
EVIDENCE_S = 5.0
SNAP_S = 1.0
TURNOVER_S = 0.4
HEADLINE = pm.SPELL_IOUS.index(pm.SPELL_IOU_HEADLINE)
ORACLES = ("decoder", "snap edges", "turnovers", "dead ball", "both")
GROUPS = (("idsse", "harmonized"), ("idsse", "raw"), ("val_metrica", "harmonized"))


# -- decoding with time-varying costs (the oracles) ---------------------------------------------

def viterbi_tv(logp, trans):
    """Best path for one cost setting whose transition matrix can change at every step. logp (n, 3),
    trans (n, 3, 3): log cost of moving from -> to on entering step t. Same tie-breaking as
    decoder.viterbi, so equal matrices give the same path."""
    n = len(logp)
    back = np.empty((n, 3), dtype=np.int8)
    delta = logp[0].copy()
    cols = np.arange(3)
    for t in range(1, n):
        cand = delta[:, None] + trans[t]
        back[t] = cand.argmax(axis=0)
        delta = cand[back[t], cols] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_tv(proba, halves, tags, setting, block, free_switch=None, change=None):
    """decoder.decode for one setting (with its change signal, if any), with the switch cost set to
    0 on the blocks that touch a frame where free_switch is True."""
    n = len(proba)
    out = np.full(n, "X", dtype="<U1")
    zone = tag_zone(tags, n, setting["tag_margin_s"])
    free = np.zeros(n, dtype=bool) if free_switch is None else free_switch
    logp = np.log(np.clip(proba, MIN_P, 1.0))
    sw, td, fd, ft = (float(setting[c]) for c in COSTS)
    k = float(setting.get("change_k", 0))

    def blocks(x, nb, pad, fill=0):
        return np.concatenate([x, np.full((pad,) + x.shape[1:], fill, dtype=x.dtype)]).reshape(nb, block, *x.shape[1:])

    for a, b in halves:
        m = b + 1 - a
        nb = -(-m // block)
        pad = nb * block - m
        lp = blocks(logp[a:b + 1], nb, pad).sum(axis=1)
        zb = blocks(zone[a:b + 1], nb, pad).any(axis=1)
        fb = blocks(free[a:b + 1], nb, pad).any(axis=1)
        if change is None:
            swb = np.full(nb, sw)
        else:
            swb = np.maximum(sw - k * blocks(change[a:b + 1], nb, pad, -np.inf).max(axis=1), 0.0)
        trans = np.zeros((nb, 3, 3))
        trans[:, 0, 1] = trans[:, 1, 0] = np.where(fb, 0.0, -swb)
        trans[:, 0, 2] = trans[:, 1, 2] = -td
        trans[:, 2, 0] = trans[:, 2, 1] = np.where(zb, -ft, -fd)
        out[a:b + 1] = STATES[np.repeat(viterbi_tv(lp, trans), block)[:m]]
    return out


def dead_oracle(proba, gt):
    """The classifier's probabilities with the true alive / dead state forced in: p_D = 1 on dead
    frames, p_D = 0 on live ones (H and A renormalised, so the team call stays the classifier's)."""
    q = proba.copy()
    dead, live = gt == "D", np.isin(gt, ("H", "A"))
    q[dead] = (0.0, 0.0, 1.0)
    team = q[live, :2] + 1e-9
    q[live, :2] = team / team.sum(axis=1, keepdims=True)
    q[live, 2] = 0.0
    return q


def turnover_mask(gt_codes, runs):
    """True within TURNOVER_S of every true change of team (H <-> A, no stoppage in between)."""
    mask = np.zeros(len(gt_codes), dtype=bool)
    r = int(round(TURNOVER_S * FPS))
    for f, src, dst in pm.transitions(gt_codes, runs):
        if src in pm.POSSESSION_LABELS and dst in pm.POSSESSION_LABELS:
            mask[max(f - r, 0):f + r + 1] = True
    return mask


def snap(pred_codes, gt_codes, runs):
    """Every decoded transition moved to the nearest true transition into the same label within
    SNAP_S; a segment whose edges would cross is dropped (merged into the one before)."""
    tol = int(round(SNAP_S * FPS))
    targets = {}
    for f, _, dst in pm.transitions(gt_codes, runs):
        targets.setdefault(dst, []).append(f)
    targets = {k: np.array(v) for k, v in targets.items()}
    out = pred_codes.copy()
    for a, b in runs:
        segs = pm.segments(pred_codes, [(a, b)])
        starts, codes = [segs[0][0]], [segs[0][2]]
        for s, _, k in segs[1:]:
            new = s
            cand = targets.get(k)
            if cand is not None and len(cand):
                near = cand[np.argmin(np.abs(cand - s))]
                if abs(near - s) <= tol:
                    new = int(near)
            if new <= starts[-1]:
                new = s if s > starts[-1] else None
            if new is None or k == codes[-1]:
                continue
            starts.append(new)
            codes.append(k)
        for s, e, k in zip(starts, starts[1:] + [b], codes):
            out[s:e] = k
    return out


# -- the possession tables ----------------------------------------------------------------------

def greedy(overlaps, threshold):
    """{true index: decoded index}: the official one-to-one greedy match (pm.matched_gt_flags),
    returned with its pairs."""
    pairs, used = {}, set()
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if i in pairs or j in used:
            continue
        pairs[i] = j
        used.add(j)
    return pairs


def bucket(value, edges):
    for lo, hi in edges:
        if lo <= value < hi:
            return "%g-%gs" % (lo, hi) if np.isfinite(hi) else ">=%gs" % lo
    return "?"


def ball_bucket(value):
    for lo, hi in BALL:
        if lo <= value < hi:
            return "%d-%d%%" % (100 * lo, min(100 * hi, 100))
    return "?"


def spell_rows(gt_codes, pred_codes, logp, visible, runs):
    """One row per true possession and one per decoded possession."""
    gt_sp = pm.possession_spells(gt_codes, runs)
    pr_sp = pm.possession_spells(pred_codes, runs)
    overlaps = pm.spell_overlaps(gt_sp, pr_sp)
    pairs = []
    for threshold in pm.SPELL_IOUS:
        g = greedy(overlaps, threshold)
        official = pm.matched_gt_flags(overlaps, len(gt_sp), threshold)
        assert np.array_equal(np.isin(np.arange(len(gt_sp)), list(g)), official)
        pairs.append(g)
    taken = [set(g.values()) for g in pairs]
    head = pairs[HEADLINE]
    over_g, over_p = [[] for _ in gt_sp], [[] for _ in pr_sp]
    for iou, i, j in overlaps:
        over_g[i].append((iou, j))
        over_p[j].append((iou, i))

    context, seg_of = {}, {}
    for a, b in runs:
        segs = pm.segments(gt_codes, [(a, b)])
        for k, (s, e, c) in enumerate(segs):
            context[s] = (NAMES[segs[k - 1][2]] if k else "start", NAMES[segs[k + 1][2]] if k + 1 < len(segs) else "end",
                          segs[k - 1][0] if k else s, segs[k + 1][1] if k + 1 < len(segs) else e)

    n = len(gt_codes)
    idx = np.flatnonzero((gt_codes < 3) & (pred_codes < 3))
    contrib = np.zeros(n)
    contrib[idx] = logp[idx, gt_codes[idx]] - logp[idx, pred_codes[idx]]
    cum = np.concatenate([[0.0], np.cumsum(contrib)])
    right = np.concatenate([[0], np.cumsum(logp.argmax(axis=1) == gt_codes)])
    seen = np.concatenate([[0], np.cumsum(visible)])
    reach = int(round(EVIDENCE_S * FPS))
    kind = {"D": "restart", "H": "turnover", "A": "turnover", "start": "half start", "end": "half end"}

    true_rows = []
    for i, (s, e, c) in enumerate(gt_sp):
        prev, nxt, lo, hi = context[s]
        lo, hi = max(lo, s - reach), min(hi, e + reach)
        row = {"side": "true", "start": s, "stop": e, "team": NAMES[c], "dur_s": (e - s) / FPS,
               "start_type": kind[prev], "end_type": "stoppage" if nxt == "D" else kind[nxt],
               "hits": sum(i in g for g in pairs), "matched": i in head,
               "ball": (seen[e] - seen[s]) / (e - s), "ball_at_start": float(visible[max(s - 12, 0):s + 13].mean()),
               "evidence": cum[hi] - cum[lo], "evidence_in": cum[e] - cum[s], "clf_right": (right[e] - right[s]) / (e - s),
               "best_iou": 0.0, "start_lag": None, "end_lag": None, "mode": "", "detail": ""}
        if over_g[i]:
            iou, j = max(over_g[i])
            row["best_iou"] = iou
        if i in head:
            ps, pe, _ = pr_sp[head[i]]
            row["start_lag"], row["end_lag"] = ps - s, pe - e
        elif not over_g[i]:
            inside = pred_codes[s:e]
            row["mode"] = "absent"
            row["detail"] = "as rival" if np.sum(inside == 1 - c) >= np.sum(inside == pm.D) else "as dead"
        elif len(over_p[j]) >= 2:
            ps, pe, _ = pr_sp[j]
            row["mode"] = "merged"
            row["detail"] = "over a stoppage" if np.any(gt_codes[ps:pe] == pm.D) else "over a rival spell"
        elif len(over_g[i]) >= 2:
            row["mode"] = "split"
        else:
            row["mode"] = "shifted"
        true_rows.append(row)

    pred_rows = []
    for j, (s, e, c) in enumerate(pr_sp):
        row = {"side": "decoded", "start": s, "stop": e, "team": NAMES[c], "dur_s": (e - s) / FPS,
               "hits": sum(j in t for t in taken[:]), "matched": j in taken[HEADLINE], "mode": "", "detail": ""}
        if not row["matched"]:
            if not over_p[j]:
                under = gt_codes[s:e]
                row["mode"] = "invented"
                row["detail"] = "over rival" if np.sum(under == 1 - c) >= np.sum(under == pm.D) else "over dead"
            elif len(over_p[j]) >= 2:
                row["mode"] = "merged"
            elif over_p[j][0][1] in head:
                row["mode"] = "fragment"
            else:
                row["mode"] = "shifted"
        pred_rows.append(row)
    return true_rows, pred_rows


# -- summaries ----------------------------------------------------------------------------------

def table(rows, key, order=None):
    """{group: n, share, recall (mean over the IoU thresholds), share of all misses}."""
    misses = sum(5 - r["hits"] for r in rows) or 1
    out = {}
    for g in order or sorted({key(r) for r in rows}):
        sub = [r for r in rows if key(r) == g]
        if sub:
            out[g] = {"n": len(sub), "share": len(sub) / len(rows), "rate": float(np.mean([r["hits"] for r in sub])) / 5,
                      "share_of_misses": sum(5 - r["hits"] for r in sub) / misses}
    return out


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def summarize(true_rows, pred_rows):
    dur_order = [bucket(lo, DURATIONS) for lo, _ in DURATIONS]
    ball_order = [ball_bucket(lo) for lo, _ in BALL]
    missed = [r for r in true_rows if not r["matched"]]
    spurious = [r for r in pred_rows if not r["matched"]]
    out = {
        "n_true": len(true_rows), "n_decoded": len(pred_rows),
        "recall": float(np.mean([r["hits"] for r in true_rows])) / 5,
        "precision": float(np.mean([r["hits"] for r in pred_rows])) / 5 if pred_rows else 0.0,
        "true_by_duration": table(true_rows, lambda r: bucket(r["dur_s"], DURATIONS), dur_order),
        "decoded_by_duration": table(pred_rows, lambda r: bucket(r["dur_s"], DURATIONS), dur_order),
        "true_by_start": table(true_rows, lambda r: r["start_type"], ("restart", "turnover", "half start")),
        "true_by_end": table(true_rows, lambda r: r["end_type"], ("stoppage", "turnover", "half end")),
        "true_by_ball": table(true_rows, lambda r: ball_bucket(r["ball"]), ball_order),
        "missed": {"n": len(missed), "modes": {}, "blame": {}},
        "spurious": {"n": len(spurious), "modes": {}},
        "lags": {},
    }
    for m in ("absent", "merged", "split", "shifted"):
        sub = [r for r in missed if r["mode"] == m]
        out["missed"]["modes"][m] = {"n": len(sub), "share": len(sub) / max(len(missed), 1),
                                     "detail": {d: sum(r["detail"] == d for r in sub) for d in sorted({r["detail"] for r in sub})}}
    for m in ("invented", "merged", "fragment", "shifted"):
        sub = [r for r in spurious if r["mode"] == m]
        out["spurious"]["modes"][m] = {"n": len(sub), "share": len(sub) / max(len(spurious), 1),
                                       "detail": {d: sum(r["detail"] == d for r in sub) for d in sorted({r["detail"] for r in sub})}}
    ev = np.array([r["evidence"] for r in missed])
    out["missed"]["blame"] = {
        "classifier_agrees_with_decoder": float(np.mean(ev <= 0)) if len(ev) else None,
        "classifier_saw_it": float(np.mean(ev > 0)) if len(ev) else None,
        "classifier_saw_it_inside": float(np.mean([r["evidence_in"] > 0 for r in missed])) if missed else None,
        "median_evidence": pct(ev, 50),
        "median_clf_right_missed": pct([r["clf_right"] for r in missed], 50),
        "median_clf_right_matched": pct([r["clf_right"] for r in true_rows if r["matched"]], 50),
        "by_mode": {m: float(np.mean([r["evidence"] > 0 for r in missed if r["mode"] == m]))
                    for m in ("absent", "merged", "split", "shifted") if any(r["mode"] == m for r in missed)},
    }
    matched = [r for r in true_rows if r["matched"]]
    for edge, key, types in (("start", "start_type", ("restart", "turnover")), ("end", "end_type", ("stoppage", "turnover"))):
        for t in types:
            lags = [r[edge + "_lag"] for r in matched if r[key] == t]
            out["lags"]["%s after %s" % (edge, t) if edge == "start" else "%s at %s" % (edge, t)] = {
                "n": len(lags), "median": pct(lags, 50), "p10": pct(lags, 10), "p90": pct(lags, 90),
                "mean_abs": float(np.mean(np.abs(lags))) if lags else None}
    return out


def aggregate(rows, bench):
    keys = ("spell_f1", "macro_f1", "possessions")
    return (by_match if bench == "idsse" else mean_of)(rows, keys)


def show(name, s, oracles):
    print("\n== %s: %d true possessions, %d decoded ==" % (name, s["n_true"], s["n_decoded"]))
    print("  recall %.3f   precision %.3f   (mean over the IoU thresholds, pooled over sequences)" % (
        s["recall"], s["precision"]))
    print("  %-14s %6s %6s %8s %8s %16s" % ("duration", "true", "decod.", "recall", "precis.", "share of misses"))
    for g, t in s["true_by_duration"].items():
        d = s["decoded_by_duration"].get(g, {"n": 0, "rate": 0.0})
        print("  %-14s %6d %6d %8.3f %8.3f %15.0f%%" % (g, t["n"], d["n"], t["rate"], d["rate"], 100 * t["share_of_misses"]))
    for title, tab in (("starts with", "true_by_start"), ("ends with", "true_by_end"), ("ball visible", "true_by_ball")):
        print("  %-14s %s" % (title, "   ".join("%s %d (recall %.3f, %.0f%% of misses)" % (
            g, t["n"], t["rate"], 100 * t["share_of_misses"]) for g, t in s[tab].items())))
    mi = s["missed"]
    print("  missed at IoU %.1f: %d   %s" % (pm.SPELL_IOU_HEADLINE, mi["n"], "   ".join("%s %.0f%%%s" % (
        m, 100 * v["share"], (" (" + ", ".join("%s %d" % kv for kv in v["detail"].items()) + ")") if v["detail"] else "")
        for m, v in mi["modes"].items())))
    b = mi["blame"]
    if b["median_evidence"] is not None:
        print("  blame of the missed: classifier agrees with the decoder %.0f%%, classifier saw it %.0f%% (median "
              "evidence %.1f; by mode %s; %.0f%% counting the possession's own frames only); classifier right on %.0f%% of the frames of a missed possession vs %.0f%% "
              "of a matched one (medians)" % (
                  100 * b["classifier_agrees_with_decoder"], 100 * b["classifier_saw_it"], b["median_evidence"],
                  ", ".join("%s %.0f%%" % (m, 100 * v) for m, v in b["by_mode"].items()),
                  100 * b["classifier_saw_it_inside"],
                  100 * b["median_clf_right_missed"], 100 * b["median_clf_right_matched"]))
    sp = s["spurious"]
    print("  spurious decoded: %d   %s" % (sp["n"], "   ".join("%s %.0f%%%s" % (
        m, 100 * v["share"], (" (" + ", ".join("%s %d" % kv for kv in v["detail"].items()) + ")") if v["detail"] else "")
        for m, v in sp["modes"].items())))
    print("  edge lag of matched possessions, decoded - true, frames: %s" % "   ".join(
        "%s median %+.0f [p10 %+.0f, p90 %+.0f]" % (k, v["median"], v["p10"], v["p90"])
        for k, v in s["lags"].items() if v["n"]))
    print("  %-12s %8s %8s %12s" % ("oracle", "spellF1", "macroF1", "possessions"))
    for o in ORACLES:
        r = oracles[o]
        print("  %-12s %8.3f %8.3f %12.0f" % (o, r["spell_f1"], r["macro_f1"], r["possessions"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", default=None, help="a train_classifier.py output folder with a fitted decoder; "
                                                     "default: the candidate's")
    parser.add_argument("--datasets", default=None, help="default: the folder the predictions were made on")
    parser.add_argument("--fit-on", choices=("both", "idsse", "val_metrica"), default="idsse",
                        help="which fitted decoder to analyse (decoder.py --fit-on; default: idsse, the candidate)")
    parser.add_argument("--target", choices=VERSIONS, default="harmonized",
                        help="which fitted decoder to analyse (decoder.py --target)")
    parser.add_argument("--change", default=None, help="which fitted decoder to analyse (decoder.py --change)")
    args = parser.parse_args()
    args.pred = args.pred or candidate_predictions()

    _, datasets = pred_datasets(args.pred, args.datasets)
    path = decoder_file(args.pred, args.fit_on, args.target, args.change)
    if not os.path.isfile(path):
        raise SystemExit("%s not found: run src/mtah/decoder.py --pred %s first" % (path, args.pred))
    with open(path) as handle:
        fitted = json.load(handle)
    setting, block = fitted["params"], fitted["block"]
    print("datasets %s\ndecoder %s: %s, block %d" % (datasets, os.path.basename(path),
                                                     ", ".join("%s %g" % kv for kv in setting.items()), block))
    t0 = time.time()

    import build_datasets as bd
    items = load_bench(args.pred, datasets, args.change)
    for (bench, match, seed), it in items.items():
        it["visible"] = ~np.isnan(bd.load_match(datasets, match, seed)["ball"][:, 0])
        it["pred"] = decode(it["p"], it["halves"], it["tags"], [[setting[c] for c in COSTS]],
                            setting["tag_margin_s"], block, it.get("change"), [setting.get("change_k", 0)])[0]
        tv = decode_tv(it["p"], it["halves"], it["tags"], setting, block, change=it.get("change"))
        assert np.array_equal(tv, it["pred"]), "decode_tv does not reproduce decoder.decode"

    report, csv_rows = {"datasets": datasets, "decoder": path, "params": setting, "block": block, "groups": {}}, []
    for bench, version in GROUPS:
        keys = sorted(k for k in items if k[0] == bench)
        true_all, pred_all, scores = [], [], {o: [] for o in ORACLES}
        for key in keys:
            it = items[key]
            gt = it["gt"][version]
            gt_codes = pm.label_codes(gt)
            runs = pm.valid_runs(gt_codes)
            pred_codes = pm.fill_abstentions(pm.label_codes(it["pred"]), runs)
            logp = np.log(np.clip(it["p"], MIN_P, 1.0))
            t_rows, p_rows = spell_rows(gt_codes, pred_codes, logp, it["visible"], runs)
            for r in t_rows + p_rows:
                r.update(bench=bench, version=version, match=key[1], seed=key[2])
            true_all += t_rows
            pred_all += p_rows

            free = turnover_mask(gt_codes, runs)
            dead_p = dead_oracle(it["p"], gt)
            variants = {
                "decoder": it["pred"],
                "snap edges": NAMES[snap(pred_codes, gt_codes, runs)],
                "turnovers": decode_tv(it["p"], it["halves"], it["tags"], setting, block, free, it.get("change")),
                "dead ball": decode_tv(dead_p, it["halves"], it["tags"], setting, block, None, it.get("change")),
                "both": decode_tv(dead_p, it["halves"], it["tags"], setting, block, free, it.get("change")),
            }
            for o, labels in variants.items():
                scores[o].append(dict(score(gt, labels, it["graded"]), match=key[1], seed=key[2]))
            # the rows must add up to the official spellF1 (a match at an IoU holds at every lower one)
            tp = np.array([[r["hits"] > k for r in t_rows] for k in range(len(pm.SPELL_IOUS))]).sum(axis=1)
            f1 = np.mean(2 * tp / (len(t_rows) + len(p_rows)))
            assert abs(f1 - scores["decoder"][-1]["spell_f1"]) < 1e-9, (f1, scores["decoder"][-1]["spell_f1"])
        name = "%s / %s labels" % (bench, version) if bench == "idsse" else "%s / its own labels" % bench
        summary = summarize(true_all, pred_all)
        summary["oracles"] = {o: aggregate(rows, bench) for o, rows in scores.items()}
        report["groups"][name] = summary
        show(name, summary, summary["oracles"])
        csv_rows += true_all + pred_all

    stem = path[:-len(".json")] + "_analysis"            # one pair of files per fitted decoder
    out = stem + ".json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    fields = ("bench", "version", "match", "seed", "side", "team", "start", "stop", "dur_s", "start_type", "end_type",
              "hits", "matched", "mode", "detail", "best_iou", "start_lag", "end_lag", "ball", "ball_at_start",
              "evidence", "evidence_in", "clf_right")
    with open(stem + "_spells.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print("\nsaved %s and %s_spells.csv  (%.0f s)" % (out, os.path.basename(stem), time.time() - t0))


if __name__ == "__main__":
    main()
