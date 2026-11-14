#!/usr/bin/env python3
"""The one command that regenerates the submission: train the final model, predict the match.

    python tools/build_datasets.py --verify                         # the public datasets, once
    python scripts/make_submission.py --data-dir "$MTAH_DATA_DIR"   # --train, then --predict

    python scripts/make_submission.py --train                       # only the model
    python scripts/make_submission.py --predict --data-dir fake_data/seed0
    python scripts/make_submission.py --predict --data-dir ... --check-public-5min

--train fits the candidate (src/mtah/features.py MODEL_GROUPS, the classifier's default parameters,
the original labels) on every IDSSE train + val match and seed, that is, the union of the folds of
src/mtah/train_classifier.py, sampled the same way. It writes external/models/<name>/model.joblib
and manifest.json (commit, datasets, groups, labels, library versions, hashes).

--predict reads the match with the readers tests/test_challenge_roundtrip.py checks
(tools/heuristic_pipeline.load_tracking and starter/load_smart_tagging), computes frame_features(),
predicts, decodes with the costs in scripts/decoder_idsse.json and writes frame,label for every
video frame: X outside the graded halves and D between the start of the graded second half and its
kick-off. The file goes through evaluation/validate_submission.py.

The file is written next to the data, in <data-dir>/submission/<team>.csv, so a file built from the
real match never lands in the repository.

REAL DATA: the console prints aggregates only (rows per label, possessions per half, % D, possession
lengths) and an error never prints data unless --debug is given, which is for fake data. Paste the
console output, never anything else from the data folder.

--check-public-5min adds spellF1 and macroF1 of the file and of smooth50 on the 5 public minutes. It
is a check that the pipeline is aligned: RULES.md forbids choosing anything with it.

EXPERIMENTAL, off by default: --change adds the turnover detector (src/mtah/change_detector.py), which
makes the decoder's switch cost depend on the input (src/mtah/decoder.py --change). --train --change
also fits the detector on the same IDSSE matches and seeds (harmonized labels, +-0.4 s);
--predict --change decodes with scripts/decoder_idsse_change.json and writes to
<data-dir>/submission_change, so the candidate's file is never overwritten. Without --change
nothing differs from the candidate.

Exact reproduction: fixed seeds, pinned libraries (requirements-mtah.txt) and one OpenMP thread,
because the boosting sums gradients in parallel and the number of threads moves the last bits of the
model between machines. scripts/decoder_idsse.json and scripts/bank_reference.json are small and
live in the repository; --bank-reference refreshes the second from a train_classifier.py run.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"                 # before numpy and sklearn are imported

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "src", "tools", "evaluation", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import heuristic_pipeline as hp  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from mtah import change_detector as cd  # noqa: E402
from mtah import decoder  # noqa: E402
from mtah import train_classifier as tc  # noqa: E402
from mtah.features import COLUMNS, MODEL_GROUPS, frame_features  # noqa: E402

FPS = tc.FPS
LABELS = "raw"
STEP = 5                                            # train on one frame in STEP, per match
HYPERPARAMETERS = tc.HYPERPARAMETERS                # defined once, in src/mtah/train_classifier.py
MODEL_NAME = "gbm_%s_%s" % (LABELS, "-".join(MODEL_GROUPS))
CV_NAME = "cv_" + MODEL_NAME
DATASETS = os.path.join(ROOT, "external", "datasets_v2_1")
DATASETS_VERSION = "2.1"
MODELS = os.path.join(ROOT, "external", "models")
PREDICTIONS = os.path.join(ROOT, "external", "predictions_v2_1")
COSTS_FILE = os.path.join(HERE, "decoder_idsse.json")
REFERENCE_FILE = os.path.join(HERE, "bank_reference.json")
# --change (experimental): the turnover detector, as src/mtah/change_detector.py runs it by default
CHANGE_GROUPS = cd.DEFAULT_GROUPS
CHANGE_LABELS = "harmonized"
CHANGE_RADIUS_S = 0.4
CHANGE_NAME = "change_gbm_%s_r%g" % (CHANGE_LABELS, CHANGE_RADIUS_S)
CHANGE_COSTS_FILE = os.path.join(HERE, "decoder_idsse_change.json")
CHANGE_REFERENCE_FILE = os.path.join(HERE, "bank_reference_change.json")
LIBRARIES = ("numpy", "pandas", "sklearn", "scipy", "joblib")
HOME_TAG, AWAY_TAG, NO_TEAM_TAG = "Barcelona", "Atletic Club", "N/A"      # as in the Smart Tagging file
CHECK_WINDOW_S = (5.0, 15.0)                        # after a window starts: the side taking the restart is near the ball


# ==================================================================================================
# The model
# ==================================================================================================

def versions():
    import importlib
    out = {name: importlib.import_module(name).__version__ for name in LIBRARIES}
    out["python"] = platform.python_version()
    return out


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def columns():
    return [c for g in MODEL_GROUPS for c in COLUMNS[g]]


def training_set(datasets):
    """(features, labels, [(match, seed)]) sampled as src/mtah/train_classifier.py samples a fold:
    every match weighs the same, whatever its number of seeds."""
    import pandas as pd
    with open(os.path.join(datasets, "splits.json")) as handle:
        splits = json.load(handle)
    pool = [tuple(x) for split in tc.CV_SPLITS for x in splits[split]]
    cols = columns()
    xs, ys = [], []
    for match in sorted({m for m, _ in pool}):
        seeds = [s for m, s in pool if m == match]
        for seed in seeds:
            d = tc.load(datasets, match, seed, groups=MODEL_GROUPS)
            keep = np.flatnonzero(d["graded"] & np.isin(d["labels"][LABELS], tc.CLASSES))[::STEP * len(seeds)]
            xs.append(d["features"].iloc[keep][cols])
            ys.append(d["labels"][LABELS][keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), pool


def train(args):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    if not os.path.isfile(os.path.join(args.datasets, "splits.json")):
        raise SystemExit("no datasets in %s: build them with python tools/build_datasets.py --verify" % args.datasets)
    t0 = time.time()
    x, y, pool = training_set(args.datasets)
    print("training set: %d frames from %d match x seed sequences  (%.0f s)" % (len(x), len(pool), time.time() - t0))
    model = HistGradientBoostingClassifier(**HYPERPARAMETERS)
    model.fit(x, y)
    folder = os.path.join(args.models_dir, MODEL_NAME)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model.joblib")
    joblib.dump(model, path, compress=3)
    data = hashlib.sha256(x.to_numpy(np.float32).tobytes() + "".join(y.tolist()).encode()).hexdigest()
    manifest = {"name": MODEL_NAME, "commit": tc.git_commit(), "datasets": os.path.basename(os.path.normpath(args.datasets)),
                "datasets_version": DATASETS_VERSION, "groups": list(MODEL_GROUPS), "columns": columns(), "labels": LABELS,
                "step": STEP, "hyperparameters": HYPERPARAMETERS, "sequences": [list(p) for p in pool],
                "training_frames": len(x), "training_data_sha256": data, "model_sha256": sha256_file(path),
                "libraries": versions(), "omp_num_threads": os.environ["OMP_NUM_THREADS"]}
    with open(os.path.join(folder, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("model in %s  (%.0f s)\n  model sha256 %s\n  data  sha256 %s" % (folder, time.time() - t0, manifest["model_sha256"], data))
    if args.change:
        train_change(args)


def change_columns():
    return [c for g in CHANGE_GROUPS for c in COLUMNS[g]]


def change_training_set(datasets):
    """As training_set, for the turnover detector: its feature groups, its target (a direct H <-> A
    change within CHANGE_RADIUS_S on the CHANGE_LABELS) and the base rate the decoder centres its
    log-odds on, over every scored frame of every sequence, as src/mtah/change_detector.py computes it."""
    import pandas as pd
    with open(os.path.join(datasets, "splits.json")) as handle:
        splits = json.load(handle)
    pool = [tuple(x) for split in tc.CV_SPLITS for x in splits[split]]
    cols, radius = change_columns(), int(round(CHANGE_RADIUS_S * FPS))
    xs, ys, every = [], [], []
    for match in sorted({m for m, _ in pool}):
        seeds = [s for m, s in pool if m == match]
        for seed in seeds:
            d = tc.load(datasets, match, seed, groups=CHANGE_GROUPS)
            labels = d["labels"][CHANGE_LABELS]
            scored = d["graded"] & np.isin(labels, tc.CLASSES)
            target = cd.near_change(labels, radius)
            keep = np.flatnonzero(scored)[::STEP * len(seeds)]
            xs.append(d["features"].iloc[keep][cols])
            ys.append(target[keep])
            every.append(target[scored])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), pool, float(np.mean(np.concatenate(every)))


def train_change(args):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    x, y, pool, base = change_training_set(args.datasets)
    print("turnover detector training set: %d frames, %.1f%% positive  (%.0f s)" % (len(x), 100 * y.mean(), time.time() - t0))
    model = HistGradientBoostingClassifier(**HYPERPARAMETERS)
    model.fit(x, y)
    folder = os.path.join(args.models_dir, CHANGE_NAME)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model.joblib")
    joblib.dump(model, path, compress=3)
    data = hashlib.sha256(x.to_numpy(np.float32).tobytes() + y.astype(np.uint8).tobytes()).hexdigest()
    manifest = {"name": CHANGE_NAME, "commit": tc.git_commit(), "datasets": os.path.basename(os.path.normpath(args.datasets)),
                "datasets_version": DATASETS_VERSION, "groups": list(CHANGE_GROUPS), "columns": change_columns(),
                "labels": CHANGE_LABELS, "radius_s": CHANGE_RADIUS_S, "base_rate": base, "step": STEP,
                "hyperparameters": HYPERPARAMETERS, "sequences": [list(p) for p in pool], "training_frames": len(x),
                "training_data_sha256": data, "model_sha256": sha256_file(path), "libraries": versions(),
                "omp_num_threads": os.environ["OMP_NUM_THREADS"]}
    with open(os.path.join(folder, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("turnover detector in %s  (%.0f s)\n  model sha256 %s\n  data  sha256 %s" % (
        folder, time.time() - t0, manifest["model_sha256"], data))


def load_change_model(models_dir):
    """(detector, manifest); stops if it is missing or is not the one this code describes."""
    import joblib
    folder = os.path.join(models_dir, CHANGE_NAME)
    if not os.path.isfile(os.path.join(folder, "manifest.json")):
        raise SystemExit("no turnover detector in %s: run python scripts/make_submission.py --train --change" % folder)
    with open(os.path.join(folder, "manifest.json")) as handle:
        manifest = json.load(handle)
    if manifest["columns"] != change_columns() or manifest["labels"] != CHANGE_LABELS or manifest["radius_s"] != CHANGE_RADIUS_S:
        raise SystemExit("the detector in %s is not the one this code describes: run --train --change" % folder)
    return joblib.load(os.path.join(folder, "model.joblib")), manifest


def load_change_costs():
    """The decoder costs fitted on the IDSSE folds with the detector's out-of-fold signal
    (src/mtah/decoder.py --change), with the base rate its change_k is centred on."""
    with open(CHANGE_COSTS_FILE) as handle:
        costs = json.load(handle)
    if (costs["trained_on"] != LABELS or list(costs["fit_on"]) != ["idsse"] or "change_k" not in costs["params"]
            or os.path.basename(costs.get("change", "")) != CHANGE_NAME or "change_base_rate" not in costs):
        raise SystemExit("%s is not the decoder fitted on IDSSE with the %s detector" % (CHANGE_COSTS_FILE, CHANGE_NAME))
    return costs


def predict_change(model, features):
    """(n,) float32 probability of a direct change of team within CHANGE_RADIUS_S."""
    return model.predict_proba(features[change_columns()])[:, list(model.classes_).index(True)].astype(np.float32)


def load_model(models_dir):
    """(model, manifest); stops if the model is missing or is not the candidate."""
    import joblib
    folder = os.path.join(models_dir, MODEL_NAME)
    if not os.path.isfile(os.path.join(folder, "manifest.json")):
        raise SystemExit("no model in %s: run python scripts/make_submission.py --train" % folder)
    with open(os.path.join(folder, "manifest.json")) as handle:
        manifest = json.load(handle)
    if manifest["columns"] != columns() or manifest["labels"] != LABELS:
        raise SystemExit("the model in %s is not the candidate this code describes: run --train" % folder)
    now = versions()
    differ = [k for k in now if manifest["libraries"].get(k) != now[k]]
    if differ:
        print("WARNING: library versions differ from the training run (%s): see requirements-mtah.txt" % ", ".join(differ))
    return joblib.load(os.path.join(folder, "model.joblib")), manifest


def load_costs():
    """The decoder costs of the candidate, fitted on the IDSSE folds (src/mtah/decoder.py --fit-on idsse)."""
    with open(COSTS_FILE) as handle:
        costs = json.load(handle)
    if costs["trained_on"] != LABELS or list(costs["fit_on"]) != ["idsse"]:
        raise SystemExit("%s is not the decoder fitted on IDSSE for the %s-label model" % (COSTS_FILE, LABELS))
    return costs


def predict_proba(model, features):
    """(n, 3) float32 probabilities in the order of tc.CLASSES (H, A, D)."""
    order = [list(model.classes_).index(c) for c in tc.CLASSES]
    return model.predict_proba(features[columns()])[:, order].astype(np.float32)


def decode_labels(proba, halves, tags, costs, p_change=None):
    """Labels of the decoder for one match: 'X' outside `halves`. With p_change (--change), the
    switch cost follows the detector as in src/mtah/decoder.py --change."""
    params = costs["params"]
    change = ks = None
    if p_change is not None:
        change, ks = decoder.change_shift(p_change, costs["change_base_rate"]), [params["change_k"]]
    return decoder.decode(proba, halves, tags, [[params[c] for c in decoder.COSTS]], params["tag_margin_s"], costs["block"],
                          change, ks)[0]


# ==================================================================================================
# The match
# ==================================================================================================

def challenge_game():
    """The challenge match: frames, graded blocks and what the model decodes. The tracking of the
    second block starts before the kick-off; those frames are dead by definition, not decoded."""
    (_, a1, b1), (_, a2, b2) = games.half_video_blocks(games.GAME)
    kickoff = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]
    return {"n": games.source_frames("tactical", games.GAME), "pitch": tuple(games.GAMES[games.GAME]["pitch"]),
            "graded": [(a1, b1), (a2, b2)], "play": [(a1, b1), (kickoff, b2)], "forced_dead": [(a2, kickoff - 1)],
            "home_tag": HOME_TAG, "away_tag": AWAY_TAG, "xml_home": ("barcelona",), "xml_away": ("athletic", "atletic")}


def exported_game(data_dir):
    """A bank match written by tools/export_challenge_format.py, read as if it were the challenge's."""
    with open(os.path.join(data_dir, "exported_from.json")) as handle:
        origin = json.load(handle)
    with open(os.path.join(origin["datasets"], "matches", origin["match"], "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    halves = [(a, b) for _, a, b in meta["halves"]]
    return {"n": meta["n_frames"], "pitch": tuple(meta["pitch_m"]), "graded": halves, "play": halves, "forced_dead": [],
            "home_tag": meta["home"], "away_tag": meta["away"], "xml_home": (meta["home"].casefold(),),
            "xml_away": (meta["away"].casefold(),)}


def read_match(game, data_dir):
    """(players, ball, slot_team, windows, team counts): windows are the SET PIECES (lo, hi, team)
    with team 'H', 'A' or None. Stops if the teams do not map as the challenge documents."""
    from baseline import team_names
    hp.N_FRAMES, hp.L, hp.W = game["n"], game["pitch"][0], game["pitch"][1]
    paths = games.asset_paths(games.GAME, data_dir)
    players, ball, slot_team = hp.load_tracking(paths, None)
    if not (np.any(slot_team == 0) and np.any(slot_team == 1)):
        raise SystemExit("the ATD metadata does not assign slots to two teams")
    names = [str(x).casefold() for x in team_names(paths["atd_meta"])]
    home_ok = any(k in names[0] for k in game["xml_home"])
    away_ok = any(k in names[1] for k in game["xml_away"])
    print("teams: first in the ATD XML is H: %s; second is A: %s; slots H %d, A %d" % (
        "expected name" if home_ok else "UNEXPECTED NAME (mapped by order, check it)",
        "expected name" if away_ok else "UNEXPECTED NAME (mapped by order, check it)",
        int(np.sum(slot_team == 0)), int(np.sum(slot_team == 1))))
    team_of = {game["home_tag"]: "H", game["away_tag"]: "A", NO_TEAM_TAG: None}
    windows, unknown = [], 0
    for e in load_smart_tagging.load(games.GAME, data_dir):
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        tag = (e["tags"].get("Team") or NO_TEAM_TAG).strip()
        unknown += tag not in team_of
        windows.append((e["start_frame"], e["end_frame"], team_of.get(tag)))
    if unknown:
        raise SystemExit("%d SET PIECES windows have a Team tag other than the two teams or N/A" % unknown)
    return players, ball, slot_team, windows


def team_check(features, windows):
    """Label-free check of the mapping: in the seconds after a SET PIECES window starts, how often
    is a home player the one nearest the ball, in the windows tagged H and in those tagged A? H
    should be the higher: on the 12 bank sequences the difference is +0.05 to +0.32. The mapping
    itself comes from the order of the teams in the ATD XML, so this warns and does not stop."""
    near = features["nearest_home"].to_numpy()
    share = {"H": [], "A": []}
    for lo, _, team in windows:
        a, b = lo + int(CHECK_WINDOW_S[0] * FPS), lo + int(CHECK_WINDOW_S[1] * FPS)
        seen = near[max(a, 0):max(b, 0)]
        seen = seen[~np.isnan(seen)]
        if team is not None and len(seen):
            share[team].append(float(seen.mean()))
    h, a = (float(np.mean(share[t])) if share[t] else float("nan") for t in "HA")
    print("SET PIECES windows: H %d, A %d, N/A %d; a home player is nearest the ball in %.2f of the H windows, %.2f of the A ones" % (
        sum(t == "H" for _, _, t in windows), sum(t == "A" for _, _, t in windows), sum(t is None for _, _, t in windows), h, a))
    if not h > a:
        print("WARNING: not higher in the H windows, as it is on every bank match: the teams may be swapped. Check before uploading.")


# ==================================================================================================
# Aggregates (the only thing printed about a match)
# ==================================================================================================

SUMMARY_KEYS = ("share_H", "share_A", "share_D", "possessions_per_half", "possessions_per_45min", "p10_s", "p50_s", "p90_s")


def summarise(labels, play):
    """What a label sequence says over the decoded frames `play` [(first, last)]."""
    codes = pm.label_codes(labels)
    runs = [(a, b + 1) for a, b in play]
    per_run = [pm.possession_spells(codes, [run]) for run in runs]
    lengths = np.array([b - a for spells in per_run for a, b, _ in spells], dtype=np.float64) / FPS
    inside = np.concatenate([codes[a:b] for a, b in runs])
    frames = float(len(inside))
    p10, p50, p90 = np.percentile(lengths, (10, 50, 90)) if len(lengths) else (0.0, 0.0, 0.0)
    return {"share_H": float(np.sum(inside == pm.H)) / frames, "share_A": float(np.sum(inside == pm.A)) / frames,
            "share_D": float(np.sum(inside == pm.D)) / frames,
            "per_half": [len(s) for s in per_run], "possessions_per_half": float(np.mean([len(s) for s in per_run])),
            "possessions_per_45min": float(len(lengths)) / (frames / FPS / 60.0) * 45.0,
            "p10_s": float(p10), "p50_s": float(p50), "p90_s": float(p90)}


def bank_reference(args):
    """scripts/bank_reference.json: the same aggregates on the banks, from the candidate's
    out-of-fold predictions decoded with the same costs. Public data only."""
    costs = load_change_costs() if args.change else load_costs()
    pred = os.path.join(args.predictions, CV_NAME)
    items = decoder.load_bench(pred, args.datasets, os.path.join(args.predictions, CHANGE_NAME) if args.change else None)
    decoded = decoder.decode_items(items, costs["params"], costs["block"])
    rows = {b: {"decoded": [], "truth_harmonized": [], "truth_raw": []} for b in decoder.BENCHES}
    for key, it in items.items():
        bench, match, _ = key
        for name, labels in (("decoded", decoded[key]), ("truth_harmonized", it["gt"]["harmonized"]), ("truth_raw", it["gt"]["raw"])):
            rows[bench][name].append(dict(summarise(labels, it["halves"]), match=match))
    out = {"model": CV_NAME, "datasets": os.path.basename(os.path.normpath(args.datasets)), "params": costs["params"],
           "note": "Same numbers as --predict prints, on the out-of-fold predictions of the IDSSE train + val "
                   "matches and on the val_metrica ensemble. Public data only."}
    if args.change:
        out["change"] = CHANGE_NAME
    for bench, by_name in rows.items():
        out[bench] = {name: tc.by_match(r, SUMMARY_KEYS) for name, r in by_name.items()}
    path = CHANGE_REFERENCE_FILE if args.change else REFERENCE_FILE
    with open(path, "w", newline="\n") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print("bank reference in %s" % path)


def print_aggregates(labels, game, reference=REFERENCE_FILE):
    print("\nrows per label")
    codes = pm.label_codes(labels)
    for k, name in enumerate(pm.LABELS):
        print("  %s  %7d  %5.1f%%" % (name, int(np.sum(codes == k)), 100.0 * np.mean(codes == k)))
    for a, b in game["forced_dead"]:
        print("  (of the D rows, %d are the graded frames before the kick-off, %d to %d: dead by definition)" % (b + 1 - a, a, b))
    s = summarise(labels, game["play"])
    print("\npossessions per half: %s" % " / ".join(str(x) for x in s["per_half"]))
    columns_ = [("this match", s)]
    if os.path.isfile(reference):
        with open(reference) as handle:
            ref = json.load(handle)
        columns_ += [("IDSSE CV", ref["idsse"]["decoded"]), ("IDSSE truth", ref["idsse"]["truth_harmonized"]),
                     ("val_metrica", ref["val_metrica"]["decoded"]), ("Metrica truth", ref["val_metrica"]["truth_harmonized"])]
    rows = (("% H (decoded frames)", "share_H", 100.0, "%.1f"), ("% A", "share_A", 100.0, "%.1f"), ("% D", "share_D", 100.0, "%.1f"),
            ("possessions per half", "possessions_per_half", 1.0, "%.1f"), ("possessions per 45 min", "possessions_per_45min", 1.0, "%.1f"),
            ("possession length p10 (s)", "p10_s", 1.0, "%.1f"), ("p50", "p50_s", 1.0, "%.1f"), ("p90", "p90_s", 1.0, "%.1f"))
    print("\n  %-28s %s" % ("", " ".join("%13s" % name for name, _ in columns_)))
    for label, key, scale, fmt in rows:
        print("  %-28s %s" % (label, " ".join("%13s" % (fmt % (scale * c[key])) for _, c in columns_)))


def public_gt_path():
    import glob
    found = glob.glob(os.path.join(ROOT, "evaluation", "gt", "*_possession_gt_public.csv"))
    if len(found) != 1:
        raise SystemExit("expected one public ground truth in evaluation/gt/, found %d" % len(found))
    return found[0]


def check_public(labels, players, ball, slot_team, gt_path):
    """spellF1 and macroF1 of the file and of smooth50 on the public minutes: alignment only."""
    import build_datasets as bd
    from baseline import smooth
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    base = smooth(bd.nearest_player_codes({"players": players, "ball": ball, "slot_team": slot_team}), 50)
    print("\nPIPELINE CHECK on the public minutes (%d graded frames). Never choose anything with it (RULES.md)." % int(np.sum(gt != pm.X)))
    for name, pred in (("this file", labels), ("smooth50", base)):
        r = pm.evaluate(gt, pm.label_codes(pred))
        print("  %-10s spellF1 %.4f   macroF1 %.4f" % (name, r["spell_f1_mean"], r["macro_f1"]))


def predict(args):
    t0 = time.time()
    game = exported_game(args.data_dir) if args.exported else challenge_game()
    model, manifest = load_model(args.models_dir)
    costs = load_change_costs() if args.change else load_costs()
    print("model %s (commit %s, datasets %s); decoder %s" % (manifest["name"], manifest["commit"], manifest["datasets"], costs["params"]))
    if args.change:
        change_model, change_manifest = load_change_model(args.models_dir)
        print("EXPERIMENTAL --change: turnover detector %s (commit %s)" % (change_manifest["name"], change_manifest["commit"]))
    players, ball, slot_team, windows = read_match(game, args.data_dir)
    features = frame_features(players, slot_team, ball, game["pitch"], game["play"], windows, MODEL_GROUPS)
    team_check(features, windows)
    proba = predict_proba(model, features)
    p_change = None
    if args.change:
        change_features = frame_features(players, slot_team, ball, game["pitch"], game["play"], windows, CHANGE_GROUPS)
        p_change = predict_change(change_model, change_features)
    labels = decode_labels(proba, game["play"], windows, costs, p_change)
    for a, b in game["forced_dead"]:
        labels[a:b + 1] = "D"
    out_dir = args.out or os.path.join(args.data_dir, "submission_change" if args.change else "submission")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s.csv" % args.team)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (f, v) for f, v in enumerate(labels.tolist()))
    if args.save_proba:
        extra = {"p_change": p_change} if p_change is not None else {}
        np.savez_compressed(os.path.join(out_dir, "%s_proba.npz" % args.team), p=proba, classes=tc.CLASSES, **extra)
    check = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path, "--rows", str(game["n"])],
                           capture_output=True, text=True)
    print("\nevaluation/validate_submission.py: %s" % ("OK" if check.returncode == 0 else "FAILED"))
    if check.returncode != 0:
        print(check.stdout)                                 # the format problem; it quotes no match data
        raise SystemExit("the file does not pass the official validation")
    print_aggregates(labels, game, CHANGE_REFERENCE_FILE if args.change else REFERENCE_FILE)
    if args.check_public_5min:
        check_public(labels, players, ball, slot_team, args.public_gt or public_gt_path())
    print("\n%s  sha256 %s  (%.0f s; in %s)" % (os.path.basename(path), sha256_file(path), time.time() - t0,
                                                "--out" if args.out else "<data-dir>/" + os.path.basename(out_dir)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", help="fit the final model on the public datasets")
    parser.add_argument("--predict", action="store_true", help="write the submission for --data-dir")
    parser.add_argument("--bank-reference", action="store_true", help="refresh scripts/bank_reference.json")
    parser.add_argument("--data-dir", default=games.DATA_DIR, help="the match data (default: MTAH_DATA_DIR, else data/)")
    parser.add_argument("--out", default=None, help="folder of the CSV (default: <data-dir>/submission)")
    parser.add_argument("--team", default="LaMasIA", help="the CSV is <team>.csv: the team name of the registration form")
    parser.add_argument("--datasets", default=DATASETS)
    parser.add_argument("--models-dir", default=MODELS)
    parser.add_argument("--predictions", default=PREDICTIONS, help="train_classifier.py output, for --bank-reference")
    parser.add_argument("--check-public-5min", action="store_true", help="alignment check on the public minutes, never to choose")
    parser.add_argument("--public-gt", default=None, help="another ground truth for that check (fake data)")
    parser.add_argument("--exported", action="store_true", help="--data-dir is a tools/export_challenge_format.py output (tests)")
    parser.add_argument("--save-proba", action="store_true", help="also write the probabilities next to the CSV (tests)")
    parser.add_argument("--debug", action="store_true", help="show tracebacks: fake data only, they can quote the input")
    parser.add_argument("--change", action="store_true",
                        help="EXPERIMENTAL: add the turnover detector (train it with --train, decode with "
                             "scripts/decoder_idsse_change.json, write to <data-dir>/submission_change)")
    args = parser.parse_args()
    run_all = not (args.train or args.predict or args.bank_reference)
    try:
        if args.train or run_all:
            train(args)
        if args.bank_reference:
            bank_reference(args)
        if args.predict or run_all:
            predict(args)
    except Exception as error:                                         # noqa: BLE001
        if args.debug:
            raise
        # never echo the exception text: with real data it could contain values from the files
        raise SystemExit("make_submission failed (%s); no details printed on purpose" % type(error).__name__)


if __name__ == "__main__":
    main()
