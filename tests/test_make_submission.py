#!/usr/bin/env python3
"""What we deliver is what we validated. No real data: fake_data and the public banks only.

    python scripts/make_submission.py --train          # first, once
    python tests/test_make_submission.py               # about ten minutes

    python scripts/make_submission.py --train --change # EXPERIMENTAL: the turnover detector too
    python tests/test_make_submission.py --change      # the same checks on the --change path

    fake match   fake_data/seed0 end to end through the command itself: the contract of the file,
                 X outside the graded halves, D before the second-half kick-off
    determinism  two runs of the command give the same file, byte for byte (sha256)
    equivalence  val matches exported with tools/export_challenge_format.py and run through the
                 command as if they were the challenge give the probabilities, labels and scores
                 of the path the validation used (load_match). With 10 decimals of a 0-1 coordinate
                 the positions are the same numbers and the two paths must be identical, bit for
                 bit. With the 6 decimals of the default export they differ by up to 0.05 mm, which
                 flips the odd discrete feature: MAX_* below say how close is the same

The model has seen the val matches (it trains on train + val), so their scores here are not an
estimate of anything: only the difference between the two paths matters.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "src", "tools", "evaluation", "starter", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import make_submission as ms  # noqa: E402

COMMAND = os.path.join(ROOT, "scripts", "make_submission.py")
EXACT_DECIMALS = 10          # the positions read back are the same float32 numbers as the npz
MAX_LABEL_DIFF = 0.002       # with the default 6 decimals: share of frames whose label may differ
MAX_SPELL_DIFF = 0.01        # a quarter of the ~0.04 the scorer calls noise
MAX_MACRO_DIFF = 0.002


def run_command(*extra):
    env = {k: v for k, v in os.environ.items() if k != "MTAH_DATA_DIR"}
    done = subprocess.run([sys.executable, COMMAND, "--predict", "--debug"] + list(extra), capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise SystemExit("make_submission failed:\n%s\n%s" % (done.stdout[-2000:], done.stderr[-2000:]))
    return done.stdout


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_labels(path):
    import possession_metrics as pm
    return pm.read_label_csv(path)["label"].to_numpy(dtype=str)


def expect(ok, what, failures):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def fake_match(failures, data_dir, change=False):
    """The command on the fake match, twice: the file's contract and determinism."""
    import possession_metrics as pm
    work = tempfile.mkdtemp(prefix="submission_", dir=os.path.join(ROOT, "fake_data"))
    flag = ["--change"] if change else []
    try:
        gt = os.path.join(data_dir, "BAR-ATH_possession_gt_fake.csv")
        first = os.path.join(work, "first")
        out = run_command("--data-dir", data_dir, "--out", first, "--team", "fake", "--check-public-5min", "--public-gt", gt, *flag)
        print(out)
        csv = os.path.join(first, "fake.csv")
        labels = read_labels(csv)
        game = ms.challenge_game()
        expect(len(labels) == game["n"] == 199319, "fake: 199319 rows", failures)
        graded = np.zeros(game["n"], dtype=bool)
        for a, b in game["graded"]:
            graded[a:b + 1] = True
        expect(np.all(labels[~graded] == "X"), "fake: X on every frame outside 11185-90651 and 110808-190197", failures)
        expect(np.all(np.isin(labels[graded], ("H", "A", "D"))), "fake: a label on every graded frame", failures)
        expect(game["forced_dead"] == [(110808, 113741)] and np.all(labels[110808:113742] == "D"), "fake: D from 110808 to 113741", failures)
        codes = pm.label_codes(labels)
        r = pm.evaluate(pm.label_codes(pm.read_label_csv(gt)), codes)
        print("      fake match against its own truth: spellF1 %.3f   macroF1 %.3f (a simulator: says nothing about the method)" % (
            r["spell_f1_mean"], r["macro_f1"]))
        second = os.path.join(work, "second")
        run_command("--data-dir", data_dir, "--out", second, "--team", "fake", "--check-public-5min", "--public-gt", gt, *flag)
        a, b = sha256(csv), sha256(os.path.join(second, "fake.csv"))
        expect(a == b, "determinism: two runs, same file (sha256 %s)" % a[:16], failures)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def equivalence(failures, datasets, match, seed, decimals, change=False):
    """The command on an exported bank match against the path of the validation."""
    import build_datasets as bd
    from export_challenge_format import export
    from mtah import train_classifier as tc
    work = tempfile.mkdtemp(prefix="equivalence_", dir=os.path.join(ROOT, "fake_data"))
    try:
        export(datasets, match, seed, work, decimals=decimals)
        out = os.path.join(work, "out")
        print("--- %s s%d, exported with %d decimals%s" % (match, seed, decimals, ", --change" if change else ""))
        print(run_command("--data-dir", work, "--exported", "--out", out, "--team", "eq", "--save-proba",
                          *(["--change"] if change else [])))
        labels = read_labels(os.path.join(out, "eq.csv"))
        saved = np.load(os.path.join(out, "eq_proba.npz"))
        proba = saved["p"]

        model, _ = ms.load_model(ms.MODELS)
        costs = ms.load_change_costs() if change else ms.load_costs()
        m = bd.load_match(datasets, match, seed)
        meta = m["meta"]
        halves = [(a, b) for _, a, b in meta["halves"]]
        d = tc.load(datasets, match, seed, groups=ms.MODEL_GROUPS)
        direct_p = ms.predict_proba(model, d["features"])
        tags = tc.set_pieces(m["smart_tagging"], meta["home"], meta["away"])
        direct_c = None
        if change:
            detector, _ = ms.load_change_model(ms.MODELS)
            direct_c = ms.predict_change(detector, tc.load(datasets, match, seed, groups=ms.CHANGE_GROUPS)["features"])
            if decimals >= EXACT_DECIMALS:
                expect(np.array_equal(saved["p_change"], direct_c),
                       "equivalence %s s%d: identical turnover detector signal (float32, every frame)" % (match, seed), failures)
        direct = ms.decode_labels(direct_p, halves, tags, costs, direct_c)

        g = d["graded"]
        delta = np.abs(proba - direct_p)[g]
        differ = float(np.mean(labels[g] != direct[g]))
        print("      probabilities: mean |diff| %.2e, max %.2e; frames with any |diff| > 0.01: %.3f%%" % (
            delta.mean(), delta.max(), 100.0 * np.mean(delta.max(axis=1) > 0.01)))
        print("      labels: %d of %d graded frames differ (%.4f%%)" % (int(np.sum(labels[g] != direct[g])), int(g.sum()), 100 * differ))
        exact = decimals >= EXACT_DECIMALS
        if exact:
            expect(np.array_equal(proba, direct_p), "equivalence %s s%d: identical probabilities (float32, every frame)" % (match, seed), failures)
            expect(np.array_equal(labels, direct), "equivalence %s s%d: identical labels" % (match, seed), failures)
        else:
            expect(differ <= MAX_LABEL_DIFF, "equivalence %s s%d: labels within %.1f%%" % (match, seed, 100 * MAX_LABEL_DIFF), failures)
        for version, gt in d["labels"].items():
            a, b = tc.score(gt, labels, g), tc.score(gt, direct, g)
            print("      %-10s command spellF1 %.4f macroF1 %.4f | load_match spellF1 %.4f macroF1 %.4f" % (
                version, a["spell_f1"], a["macro_f1"], b["spell_f1"], b["macro_f1"]))
            limit_s, limit_m = (0.0, 0.0) if exact else (MAX_SPELL_DIFF, MAX_MACRO_DIFF)
            expect(abs(a["spell_f1"] - b["spell_f1"]) <= limit_s and abs(a["macro_f1"] - b["macro_f1"]) <= limit_m,
                   "equivalence %s s%d: scores on the %s labels within %g / %g" % (match, seed, version, limit_s, limit_m), failures)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", default=os.path.join(ROOT, "fake_data", "seed0"))
    parser.add_argument("--datasets", default=ms.DATASETS)
    parser.add_argument("--match", nargs="+", default=["J03WOY", "J03WMX"], help="val matches (seed 0)")
    parser.add_argument("--change", action="store_true",
                        help="test the EXPERIMENTAL --change path instead (after make_submission.py --train --change)")
    args = parser.parse_args()
    failures = []
    fake_match(failures, args.fake, args.change)
    for match in args.match:
        equivalence(failures, args.datasets, match, 0, EXACT_DECIMALS, args.change)
    equivalence(failures, args.datasets, args.match[0], 0, 6, args.change)
    print("\nPASSED" if not failures else "\nFAILED:\n  " + "\n  ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
