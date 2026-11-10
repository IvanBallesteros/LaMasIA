# Ball possession from tracking data: technical document

Team: LaMasIA More than a Hack 2027, Initial Challenge.

> **Draft notes (delete before submitting).** Open items, all marked `TODO` in the text:
> final decoder results (Section 6), the one-off test scores (Section 6), the organisers' reply on
> calibration and on the Metrica licence (Section 4), the team's confirmation of how the public
> minutes and the video were used (Section 4), and the team name. The byte-for-byte claim in
> Section 8 still needs a run on a second computer.

## 1. Summary

We label every frame of the match as `H`, `A`, `D` or `X` in two stages:

1. A **per-frame classifier** estimates, from the tracking data and the set-piece tags, how likely
   each of `H`, `A` and `D` is on every frame.
2. A **temporal decoder** turns those probabilities into clean, continuous possessions.

Everything that is trained or tuned uses external matches only. Nothing was fitted on the provided
game's ground truth or on its video.

## 2. Approach

### Why two stages

The ranking metric, `spellF1`, scores whole possessions and is very sensitive to when they start and
end. A frame-by-frame prediction flickers: it invents three to four times too many possessions and
scores poorly on `spellF1`, even with a good frame accuracy (Section 6). A decoder removes the
flicker without blurring the boundaries the way a smoothing window does.

### The classifier

A gradient-boosted tree model (`HistGradientBoostingClassifier` from scikit-learn) with library
default parameters and a fixed seed. We chose it for three reasons. It handles missing values
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

### The decoder

A three-state hidden Markov model (`H`, `A`, `D`), solved with the Viterbi algorithm over each half
(`src/mtah/decoder.py`). The evidence for each state is the classifier's log-probability. A path pays
a cost every time it changes state:

- `switch`: from one team to the other, without a stoppage;
- `to_dead`: the ball goes dead;
- `from_dead`: play restarts;
- `from_dead_tag`: play restarts inside a set-piece window, widened by a margin `tag_margin_s`.

It decodes blocks of 5 frames (0.2 s). The costs are chosen by coordinate descent on the official
`spellF1`, using out-of-fold predictions on the IDSSE matches. The Metrica games stay out of the
fit, so they score the decoder on unseen matches.

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
| Ball direction before and after (`turn`) | Below the noise level |
| Extra training weight near changes of possession (`--change-weight`) | Below the noise level |
| Other hyperparameters for the classifier | No gain over the defaults |

## 4. Training data

**No data from the provided game was used to train or tune anything.**

| Dataset | Matches | Labels: source | Role |
|---|---|---|---|
| IDSSE (Bassek et al., 2025): Bundesliga and 2. Bundesliga, 2022/23 | 7 (669 graded minutes) | Ball status and ball possession per frame, delivered with the DFL position data | 4 train, 2 validation, 1 test |
| Metrica Sports sample data | 3 anonymised (287 graded minutes) | Derived by us from Metrica's human-labelled events | 2 validation, 1 test |

The classifier trains on the six IDSSE train and validation matches: 171,174 frames, taking one
frame in five and giving every match the same weight. The test matches stay apart until the system
is final.

**Sources and licences.**

- IDSSE: Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data, 12. CC BY 4.0.
- Metrica Sports sample data: <https://github.com/metrica-sports/sample-data>. The repository has
  no formal licence; its README asks users to be responsible and to acknowledge the source, which
  we do here. [TODO: update with the organisers' reply]

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
the video.
[TODO: update with the organisers' reply on this calibration]

**Use of the public ground truth.** The five public minutes were used only to check alignment and
the meaning of the labels. They were never used to train, tune or choose between versions. The
output was never compared with the video. [TODO: all four of us confirm this paragraph]

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
  in control therefore does not change possession.
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

> **TODO: update this section only**, once the decoder is final.

Decoder costs: `switch` 32, `to_dead` 64, `from_dead` 32, `from_dead_tag` 0, `tag_margin_s` 2.

IDSSE scores come from cross-validation, leaving one match out each time. Metrica games were not
used for training or to fit the decoder.

| Bench | spellF1 | macroF1 | Possessions predicted / true | Baseline (smooth50): spellF1 / macroF1 |
|---|---|---|---|---|
| IDSSE, harmonised labels | 0.621 | 0.896 | 247 / 315 | 0.240 / 0.626 |
| IDSSE, original labels | 0.512 | 0.884 | 247 / 448 | 0.226 / 0.617 |
| Metrica validation games | 0.615 | 0.903 | 219 / 334 | 0.263 / 0.639 |
| Test matches, measured once | [TODO] | [TODO] | [TODO] | [TODO] |

- Without the decoder, the classifier alone gets `spellF1` 0.266 and `macroF1` 0.882 on IDSSE, with
  1,199 predicted possessions.
- On Metrica, the score above averages the six cross-validation models. A single model, which is what
  we submit, gives the same result within noise (0.619).
- With this few matches, differences in `spellF1` below about 0.04 are noise.
- These are scores on simulated tracking of other matches. They are an estimate, not a measurement
  on the provided game.

## 7. Limitations

- **Short possessions.** Possessions under 5 s are 45% of all possessions on IDSSE, and most of the
  misses. We recover 6% of those under 2 s and 44% of those between 2 and 5 s. The system predicts
  fewer possessions than there are.
- **Most of those errors come from the classifier, not the decoder.** In half of the missed
  possessions the classifier already favours the wrong team. Only in 9% does the decoder erase a
  correct prediction.
- **Changes of possession in open play are harder than restarts.** We recover 48% of the
  possessions that start right after a rival possession, against 76% of those that start after a
  dead ball.
- **Smoothed tracking.** A kick is spread over about 0.6 s, so sharp events such as touches are hard
  to see.
- **The ball is missing in 41% of the graded frames**, mostly in the hardest moments.
- **The training data is simulated to look like the challenge's.** The smoothing is an assumption,
  and the size of the ball's position error is unknown.
- **Few matches.** Six matches train the model and two more validate it.
- **Possession rules are learned from other providers' labels**, which differ from the challenge's
  ground truth.

With more time, we would detect the ball in the video during tracking gaps, look for signals of who
*controls* the ball rather than who is closest to it, and train on more matches that have video.

## 8. Reproduction

Python 3.12, with pinned dependencies. The match data is not in this repository.

```bash
pip install -r requirements-mtah.txt
python tools/build_datasets.py --verify        # downloads the public data and builds the datasets
python scripts/make_submission.py --train      # trains the final model
python scripts/make_submission.py --predict --data-dir <path to the match data> --team <team name>
```

The file is written to `<data dir>/submission/<team name>.csv` and goes through
`evaluation/validate_submission.py`.

The result is the same file, byte for byte: seeds are fixed, library versions are pinned, and
training uses a single thread so that the number of processor cores does not change the model.
`python tests/test_make_submission.py` checks the pipeline end to end on simulated data. It also
checks that the submission command gives exactly the same predictions as the path used for validation.
