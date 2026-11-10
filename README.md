# LaMasIA: ball possession from tracking data

Our solution to **More than a Hack 2027, Phase 1** (ball possession in Barça–Athletic). For every
video frame it writes `H` (home has the ball), `A` (away), `D` (dead ball) or `X` (not graded).

It works in two stages:

1. A **per-frame classifier** gives the probability of `H`, `A` and `D` on each frame, from the
   tracking data and the Smart Tagging set pieces.
2. A **decoder** (a hidden Markov model) turns those probabilities into continuous possessions.

Everything that is trained or tuned uses **other matches only** (public IDSSE and Metrica data). On
held-out matches it reaches `spellF1` ≈ 0.62, against 0.24 for the official baseline. The full
explanation is in the **[technical document](docs/technical_document.md)**.

## Installation

Python 3.12 (tested). The versions are pinned so that the file is reproduced exactly.

```bash
pip install -r requirements-mtah.txt
```

## The match data

The match data is **not** in this repository: its licence does not allow it. Put the data bundle
in `./data`, or set `MTAH_DATA_DIR` to its folder.

## Reproducing the submission

```bash
python tools/build_datasets.py --verify                         # downloads the public training data and builds the datasets (~25 min)
python scripts/make_submission.py --data-dir "$MTAH_DATA_DIR"   # trains the final model and predicts the match (~3 min)
```

The file is written to `<data dir>/submission/<team>.csv` and checked with
`evaluation/validate_submission.py`. Seeds are fixed, libraries are pinned and training uses one
thread, so the result is the same file, byte for byte.

## What is in this repository

| Folder | What |
|---|---|
| `src/mtah/` | our classifier (`features.py`, `train_classifier.py`) and decoder (`decoder.py`) |
| `scripts/make_submission.py` | the one command that regenerates the submission |
| `tools/build_datasets.py` | builds the training data from the public IDSSE and Metrica matches |
| `tests/` | tests on simulated data |
| `docs/` | technical document and team decisions (`DECISIONS.md`) |
| `experiments/log.csv` | every experiment we ran, with its scores |
| `evaluation/`, `starter/`, `games.py` | from the organisers' repository, unchanged |

## Credits and licence

- This repository starts from the organisers' challenge repository (MIT licence, see
  [LICENSE](LICENSE)). Their original README is in
  [docs/challenge_README.md](docs/challenge_README.md), and the challenge documentation
  (`CHALLENGE.md`, `DATA.md`, `RULES.md`…) is kept unchanged.
- **IDSSE**: Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data, 12. CC BY 4.0.
- **Metrica Sports sample data**: <https://github.com/metrica-sports/sample-data>
