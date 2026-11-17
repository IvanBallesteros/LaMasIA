# Ball possession from tracking data: technical document

Team LaMasIA · More than a Hack 2027, Phase 1: Ball Possession.

## 1. Summary

We label every frame of the match as `H`, `A`, `D` or `X` in two stages:

1. Two **per-frame models** read the tracking data and the set-piece tags. A **classifier**
   estimates how likely each of `H`, `A` and `D` is on every frame. A **turnover detector**
   estimates how likely the ball is to pass from one team to the other around that frame.
2. A **temporal decoder** turns the classifier's probabilities into clean, continuous possessions.
   Changing team is cheaper where the detector sees a turnover, and dearer where it does not.

Everything that is trained or tuned uses external matches only. Nothing was fitted on the provided
game's ground truth or on its video.

## 2. Approach

### Why two stages

The ranking metric, `spellF1`, scores whole possessions and is very sensitive to when they start and
end. A frame-by-frame prediction flickers: it invents three to four times too many possessions and
scores poorly on `spellF1`, even with a good frame accuracy (Section 6). A decoder removes the
flicker without blurring the boundaries the way a smoothing window does.

### The classifier

A gradient-boosted tree model (`HistGradientBoostingClassifier` from scikit-learn) with the
library's default parameters, a fixed seed and early stopping off (by default the library turns it
on with this much data). We chose it for three reasons. It handles missing values
natively, which matters because the ball is missing in 41% of the graded frames. It works well on
tabular signals with little tuning. And it can be made exactly reproducible.

It uses 71 signals per frame, in nine groups (`src/mtah/features.py`):

| Group | What it describes |
|---|---|
| `ball` | where the ball is and how fast it moves |
| `near` | which players are close to the ball now |
| `context` | who was close to the ball in the surrounding seconds |
| `edges` | who was close just before and just after this frame, kept apart |
| `players` | how fast the players move, where each team stands, whether it advances |
| `shape` | width, depth, defensive line and forward speed of each team |
| `gap` | whether the ball is seen, and how long it has been or will be missing |
| `memory` | during a gap: who was near the ball and where it was on each side of the gap |
| `tags` | Smart Tagging set-piece windows: inside one or not, tagged team, time to the start |

The `shape` group speaks from the players alone, so it still has information when the ball is
missing.

Each group was added only if it improved `spellF1` on the IDSSE matches left out in
cross-validation, with both versions of the labels, and on the Metrica games, by more than the noise
between matches (paired bootstrap). The challenge match was never used for these choices.

Compared with our first model (groups `ball`, `near`, `context`, `players`, `gap` and `tags`), adding
`memory`, `edges` and `shape` raises `spellF1` from 0.561 to 0.621 on IDSSE (harmonised labels), from
0.459 to 0.512 on IDSSE (original labels) and from 0.579 to 0.615 on Metrica. The first model's
decoder was also fitted on the Metrica games, so its Metrica score is, if anything, optimistic.

### The turnover detector

Most of the possessions we miss are short ones that start with a turnover in open play (Section 7).
A decoder with one fixed cost for changing team cannot tell where a turnover is likely, so we added a
second model for that (`src/mtah/change_detector.py`).

- **What it predicts.** For every frame, the probability that the ball passes from one team to the
  other, with no stoppage in between, within 0.4 s of that frame. This is the same 0.4 s as the
  control rule of Section 5.
- **How.** It is the same kind of model, with the same parameters, as the classifier. It uses the
  classifier's signals plus `turn`, the direction of the ball along the pitch before and after the
  frame.
- **Training.** It trains on the same six IDSSE matches, against the harmonised labels (Section 5).
  About 3% of the frames are within 0.4 s of a turnover.
- **How well it works.** Leaving one IDSSE match out at a time, it separates those frames from the
  rest with a ROC AUC of 0.88, and the same on Metrica. 74% of the turnovers fall within its top 10%
  of frames.

We kept it because it improved `spellF1` on IDSSE, with both versions of the labels, and on Metrica
(paired bootstrap, Section 6).

### The decoder

A three-state hidden Markov model (`H`, `A`, `D`), solved with the Viterbi algorithm over each half
(`src/mtah/decoder.py`). The evidence for each state is the classifier's log-probability. A path pays
a cost every time it changes state:

- `switch`: from one team to the other, without a stoppage;
- `to_dead`: the ball goes dead;
- `from_dead`: play restarts;
- `from_dead_tag`: play restarts inside a set-piece window, widened by a margin `tag_margin_s`.

The detector sets the `switch` cost at every step: `max(0, switch − k · s)`, where `s` is the
detector's log-odds that the ball changes team there, minus their average.

- Where the detector is at its average, the cost is `switch`.
- It falls to 0 where a turnover is almost certain.
- It rises to about 55 where a turnover is very unlikely.

It decodes blocks of 5 frames (0.2 s). The costs and the weight `k` are chosen together, by
coordinate descent on the official `spellF1`, using out-of-fold predictions on the IDSSE matches.
The Metrica games stay out of the fit, so they score the decoder on unseen matches.

## 3. Assets of the provided game

| Asset | Used? | How |
|---|---|---|
| Tactical automatic tracking (ATD) | Yes | Players and ball, frame by frame. The main input |
| Smart Tagging, `SET PIECES` | Yes | Restart windows and the tagged team |
| Smart Tagging, tactical phases | No | Our training data has no equivalent, so a model could not learn from them |
| Tactical video | No | We have no video for the training matches, so we could not validate its use |
| Homography, broadcast video and tracking | No | The tactical tracking already gives pitch coordinates |

Teams are mapped by order: the first team in the tracking metadata is `H`, as the documentation
states.

**Tried and dropped** (details in `docs/DECISIONS.md`):

| Idea | Result |
|---|---|
| Ball touch detector (`touches`) | No gain. On a smoothed ball, a detected touch is no more common near a change of possession than anywhere else |
| Ball direction before and after (`turn`), in the classifier | Below the noise level. The turnover detector uses it |
| Extra training weight near changes of possession (`--change-weight`) | Below the noise level |
| Other hyperparameters for the classifier | No gain over the defaults |

## 4. Training data

**No data from the provided game was used to train or tune anything.**

| Dataset | Matches | Labels: source | Role |
|---|---|---|---|
| IDSSE (Bassek et al., 2025): Bundesliga and 2. Bundesliga, 2022/23 | 7 (669 graded minutes) | Ball status and ball possession per frame, delivered with the DFL position data | 4 train, 2 validation, 1 test |
| Metrica Sports sample data | 3 anonymised (287 graded minutes) | Derived by us from Metrica's human-labelled events | 2 validation, 1 test |

The classifier and the turnover detector train on the six IDSSE train and validation matches:
171,174 frames, taking one frame in five and giving every match the same weight. The test matches
stayed apart until the system was final, and were then scored once (Section 6).

**Sources and licences.**

- IDSSE: Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data, 12. CC BY 4.0.
- Metrica Sports sample data: <https://github.com/metrica-sports/sample-data>. The repository has
  no formal licence; its README asks users to be responsible and to acknowledge the source, which
  we do here. It is published by Metrica Sports, one of the organisers of this challenge. We do not
  redistribute it: `tools/build_datasets.py` downloads it from that repository.

**Making external matches look like the challenge data.** These matches come with clean tracking and
known player identities, so we degrade them first (`tools/build_datasets.py`):

- 46 anonymous player slots, with lost players and identity switches;
- stretches without a ball, off the pitch, at stoppages and in play;
- short losses filled in a straight line, then a centred smoothing of the positions;
- a position error on the ball, whose size we vary between training seeds because it cannot be
  measured;
- synthetic set-piece windows, with the timing and the mix of types of the real file.

The settings of this degradation were calibrated on **label-free aggregate statistics** of the
provided tracking and Smart Tagging: the share of frames without a ball, the distribution of gap
lengths, where the ball was last seen before a gap, the percentiles of the positions' second
difference, and the number and timing of set-piece windows. They use neither the ground truth nor
the video. A team member ran the diagnostic (`tools/ball_gap_diagnostic.py`), which prints only
aggregates, and the values are listed in `tools/build_datasets.py` (`REAL_BALL_GAPS`, `TAG_OFFSET`,
`TAG_RECALL`). They only shape how the external matches are degraded. The models, their settings
and the decoder costs are fitted on the external matches alone.

We asked the organisers about this use. Their reply (9 November 2026) restated the clarification
email: the algorithm must not be trained, fitted or calibrated on the ground-truth possession data
or on the provided video, and must be trained and tuned on external data. We read this use as
compatible with those rules. The algorithm is trained and tuned on external data only. These
statistics contain no labels, come from neither the ground truth nor the video, and only set how
that external data is degraded, much like choosing how much noise to add. We describe it here in
full so that the jury can check it.

**Other uses of the provided game.** Apart from the prediction itself and the calibration above:

- The five public minutes were used only to check alignment and the meaning of the labels, never
  to train, tune or choose between versions.
- A second label-free diagnostic (`tools/touch_rate_diagnostic.py`) compared a few signals on the
  provided tracking and on the external matches, as a check of the simulation. No setting was taken
  from it.
- The submission command prints label-free aggregates of our output (share of each label,
  possessions per half, possession lengths) next to the same figures on the external matches. We
  used them only to catch pipeline failures, never to change anything.
- The output was never compared with the video.

**AI assistant.** We wrote code with an LLM assistant (Claude Code), under written instructions
that kept it away from the match data: the data lives outside the repository, we ran the scripts
on it ourselves, and the assistant only saw the aggregates they print, as the scripts here show.

## 5. Possession rules

We follow the definition in `CHALLENGE.md`: possession changes only when the opposing team gains
**control**, and a dead ball is a stoppage, not a possession. No rule is hand-coded for each
situation. The rules live in the labels we fit against and in the decoder's costs.

- **A touch without control, or a deflection.** The DFL channel counts these as changes. So we also
  built *harmonised* labels: a short spell of the opponent, returning to the same team without a
  stoppage, counts as a touch unless an opponent kept the ball within 1.5 m for at least 0.4 s in
  the clean tracking. The decoder's costs are fitted against these labels. The classifier trains on
  the original labels, which gave slightly better results, within the noise.
- **Loose balls and duels.** The current team keeps possession until the evidence for the other team
  is strong and sustained enough to pay the `switch` cost. A duel or a loose ball that does not end
  in control therefore does not change possession. The turnover detector makes that cost lower
  where a change of team is likely, and higher elsewhere.
- **Restarts.** Every stoppage ends in a restart with a clear possessor, the `D` to `H`/`A`
  transition. Inside a set-piece window a restart is cheaper. The team comes from the classifier,
  which sees the tagged team when the tag has one.
- **Ball out of play.** `D` is a state of its own. The ball missing is **not** treated as `D`: the
  `gap` and `memory` signals let the classifier decide.
- **Fixed by the rules of the file.** `X` outside the graded halves. `D` from frame 110808 to 113741,
  the graded frames before the second-half kick-off. No abstentions inside the graded halves.

For the Metrica games, a foul or the ball going out stops play, and only a set piece or a goal kick
restarts it. If a stoppage has no restart of its own, the first possession event restarts play.

## 6. Results

Decoder costs: `switch` 32, `to_dead` 64, `from_dead` 16, `from_dead_tag` 4, `tag_margin_s` 2, and
detector weight `k` 4.

### Validation

IDSSE scores come from cross-validation, leaving one match out each time: the classifier and the
detector that score a match never saw it. The Metrica games were not used to train the models or to
fit the decoder. They were used to decide whether to keep each part (Section 2), so their score is
slightly optimistic. Possessions are per match.

| Bench | spellF1 | macroF1 | Possessions predicted / true | Baseline (smooth50): spellF1 / macroF1 |
|---|---|---|---|---|
| IDSSE, harmonised labels | 0.627 | 0.897 | 258 / 315 | 0.240 / 0.626 |
| IDSSE, original labels | 0.526 | 0.886 | 258 / 448 | 0.226 / 0.617 |
| Metrica validation games | 0.633 | 0.904 | 230 / 334 | 0.263 / 0.639 |

### Test, measured once

The test matches, J03WR9 (IDSSE) and METRICA3, were not used for anything until the system was
final. We then scored them once with the submitted models (`scripts/evaluate_test.py`) and changed
nothing afterwards.

| Test match | spellF1 [90% interval] | macroF1 | Possessions predicted / true | Baseline (smooth50): spellF1 / macroF1 |
|---|---|---|---|---|
| J03WR9, harmonised labels | 0.532 [0.483, 0.578] | 0.858 | 237 / 282 | 0.250 / 0.656 |
| J03WR9, original labels | 0.461 [0.413, 0.508] | 0.851 | 237 / 370 | 0.231 / 0.649 |
| METRICA3 | 0.572 [0.528, 0.620] | 0.900 | 255 / 381 | 0.223 / 0.591 |

- **The test scores are lower than in validation**, by 0.06 to 0.10. Two things explain it:
  - the validation scores guided our choices, so they are somewhat optimistic;
  - matches vary: J03WR9 is below every IDSSE validation match (0.58 to 0.69).

  They are still more than twice the baseline. Of all our numbers, the test scores are the best
  estimate.
- **What the detector adds.** Against the same system without it (paired bootstrap, 95%
  intervals): +0.006 [+0.001, +0.011] on IDSSE with harmonised labels, +0.013 [+0.009, +0.017] with
  original labels and +0.018 [+0.007, +0.031] on Metrica. It wins on 14 of the 16 match and label
  pairs. The gain is small, below the noise between matches.
- **Without the decoder**, the classifier alone gets `spellF1` 0.266 and `macroF1` 0.882 on IDSSE,
  with 1,199 predicted possessions.
- **On Metrica**, the validation score averages the six cross-validation models. The single final
  model, which is what we submit, gets 0.645 (0.643 and 0.647).
- With this few matches, differences in `spellF1` below about 0.04 are noise.
- These are scores on simulated tracking of other matches. They are an estimate, not a measurement
  on the provided game.

## 7. Limitations

These figures are for IDSSE with harmonised labels (`src/mtah/decoder_analysis.py`).

- **Short possessions.** Possessions under 5 s are 45% of all possessions and three quarters of the
  misses. We recover 9% of those under 2 s and 46% of those between 2 and 5 s. The system predicts
  fewer possessions than there are.
- **Most of those errors come from the classifier, not the decoder.** In 93% of the missed
  possessions, the classifier's own evidence already prefers what the decoder output. Only in 7%
  does the decoder overrule a classifier that had seen the possession.
- **Changes of possession in open play are harder than restarts.** We recover 49% of the
  possessions that start with a turnover, against 75% of those that start after a dead ball. The
  turnover detector helps only a little: even a perfect one would raise `spellF1` from 0.63 to 0.66.
- **Lower scores on unseen matches.** The test matches score 0.06 to 0.10 below validation
  (Section 6).
- **Smoothed tracking.** A kick is spread over about 0.6 s, so sharp events such as touches are hard
  to see.
- **The ball is missing in 41% of the graded frames**, mostly in the hardest moments.
- **The training data is simulated to look like the challenge's.** The smoothing is an assumption,
  and the size of the ball's position error is unknown.
- **Few matches.** Six matches train the models, two more validate them and two test them.
- **Possession rules are learned from other providers' labels**, which differ from the challenge's
  ground truth.

With more time, we would detect the ball in the video during tracking gaps, look for signals of who
*controls* the ball rather than who is closest to it, and train on more matches that have video.

## 8. Reproduction

Python 3.12.10, with the versions pinned in `requirements-mtah.txt`. The match data is not in this
repository.

```bash
pip install -r requirements-mtah.txt
python tools/build_datasets.py --verify        # downloads the public data and builds the datasets
python scripts/make_submission.py --train      # trains the classifier and the turnover detector
python scripts/make_submission.py --predict --data-dir <path to the match data>
```

The file is written to `<data dir>/submission/LaMasIA.csv` and goes through
`evaluation/validate_submission.py`. Without `--train` or `--predict`, the command does both.

The result is the same file, byte for byte. Seeds are fixed, library versions are pinned, and
training uses a single thread, so the number of processor cores does not change the models. The
command prints these hashes, and we got the same ones on two different computers:

| Model | sha256 of the model | sha256 of its training data |
|---|---|---|
| Classifier | `2dcb9cf50821698729d775804987a160cc620cdf1cb00ad19fe3c0d08acb6579` | `bd3acaef1ad5be4c815d9be1f0b5c13fe87a38824283ee6beb8b4f285a78a3f1` |
| Turnover detector | `4271d8dd24d997ba98f094d2997a167211a0adab218a4491b30d6f0bcce29d89` | `2a9afae601006626d2ba9a13c3930d83e0c55370ae706759b9383e10b89d3ff1` |

The submitted file, `LaMasIA.csv`, has sha256
`96f80ff237074955b16053f6a862bfe8eb6f2184e765d0f8972eb38538b0141a`. The command prints it at the
end.

Two more commands check the work:

- `python tests/test_make_submission.py` checks the pipeline end to end on simulated data. It also
  checks that the submission command gives exactly the same predictions as the path used for
  validation.
- `python scripts/evaluate_test.py` reproduces the test scores of Section 6.

`--without-detector` gives the system without the turnover detector, which we do not submit.
