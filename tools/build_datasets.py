#!/usr/bin/env python3
"""Build train / validation / test datasets in the shape of the challenge, from public matches.

Two sources, two roles:

  IDSSE (Bassek, Weber, Rein & Memmert, Scientific Data 2025; CC BY 4.0) -- 7 Bundesliga and
  2. Bundesliga matches with DFL position data at 25 fps. Every ball frame carries BallStatus and
  BallPossession from corrected tracking. Plenty of data: TRAIN, VAL and TEST.
  Allowed by the organisers (metrica-sports/more-than-a-hack-2027, issue #16) if it is cited, if
  every hyperparameter and threshold is chosen on it or other games, and if the 5 public minutes
  of the provided game are only used to check the pipeline.

  Metrica Sports sample data (github.com/metrica-sports/sample-data) -- 3 anonymised matches with
  Metrica's corrected tracking and human event data. No possession channel: labels are DERIVED
  FROM THE EVENTS (see labels_from_events). Metrica's own written definition of possession matches
  CHALLENGE.md ("possession is not considered lost when an opposing player's interference does not
  have a significant impact"), so these are the closest labels to the challenge's criterion we
  can get: VAL_METRICA (games 1, 2) and TEST_METRICA (game 3).

Label harmonisation. DFL's BallPossession changes hands far more often than Metrica's criterion
(it counts a touch). Two fixed clean-ups are applied to every source -- stoppages shorter than
1 s, and possession flickers shorter than 0.2 s right after a restart, are data artefacts -- and
one CONTROL rule to IDSSE: an opponent spell that returns to the same team without a stoppage is
a touch, not a change of possession, if no opponent kept the ball within CONTROL_M metres for
CONTROL_S seconds in the clean tracking (CHALLENGE.md: "a simple touch without control", "a
deflection", "an aerial duel won without subsequent control"). A grid scored against the duration
distribution of the Metrica VAL games (label_harmonization.json keeps the grid, the loss and a
cross-check between the two games) leaves a diagonal of equivalent settings -- a shorter reach
with a shorter time or a longer reach with a longer time -- whose differences are below the noise
between the two games; within it the team picks the values on physical grounds (CONTROL_CHOICE).
Other games only, never the challenge match. The raw labels are kept beside the harmonised ones (labels_raw.csv), so every
conclusion can be checked on both.

Neither source is the challenge data: both are clean tracking with player identities. Every match
is DEGRADED to look like the challenge ATD (46 anonymous slots, identity switches, a smoothing
tracker) and a Smart Tagging file is SYNTHESISED from its restarts: SET PIECES windows of 35 s (kick-offs 5 s), as
in the real file, with the restart at a random offset inside (TAG_OFFSET). The ball follows the pattern
measured on the challenge ATD by tools/ball_gap_diagnostic.py -- aggregates only, no labels, run by
a person (REAL_BALL_GAPS): no stretch without a ball shorter than ~5 s (short losses are
interpolated) and long stretches when the ball is out (p_off), during stoppages (p_dead) and, less
often, with the ball in play (r_live). Those are calibrated against the aggregates -- shares by
stretch length, long stretches per half, SET PIECES preceded by one, and where the ball was last
seen before one -- on the IDSSE TRAIN matches. Metrica's corrected tracking has no ball during most
stoppages; a calibrated share of them (fill_dead, on the VAL_METRICA games) gets an estimated
position, the last sighting carried to the restart point. Positions come out of a model tracker:
short losses filled in a straight line, then a centred Gaussian smoother -- offline, like the ATD --
and, for the ball, a position error on the detected frames (straight through the filled ones). The
smoother widths and the second difference of the ball's error are calibrated against the
percentiles of the second difference measured on the challenge ATD (REAL_BALL_GAPS); the size of
the ball's error is not identifiable from them and varies between training seeds (TRAIN_BALL_ERROR).
ball_degradation.json keeps every grid.

    python tools/build_datasets.py --verify          # download, build v2.1 and check
    python tools/build_datasets.py --selftest        # fake match (as IDSSE) + Metrica, no figshare needed

Defaults: IDSSE raw XML in external/idsse_raw/, Metrica raw files in external/metrica_raw/, the
datasets in external/datasets_v2_1/ (external/datasets/ is v1, from tag v1-ball-degradation;
external/datasets_v2/ is v2, built by commit 34dff27). Public data, not the challenge's: fine for
Claude Code to read.

Output (see the README.md written into it):

    README.md  splits.json  summary.csv  label_harmonization.json  ball_degradation.json
    matches/<MATCH>/meta.json            source, timeline, label rule, ground-truth statistics
    matches/<MATCH>/labels.csv           frame,label (H/A/D/X), harmonised -- the official scorer's format
    matches/<MATCH>/labels_raw.csv       the labels before harmonisation
    matches/<MATCH>/truth.npz            the clean tracking, for oracle experiments and debugging
    matches/<MATCH>/seed<k>/tracking.npz ATD-like degraded tracking: 46 slots + ball
    matches/<MATCH>/seed<k>/smart_tagging.xml   synthetic, SportsCode XML like the real file
    matches/<MATCH>/seed<k>/degradation.json    parameters and the resulting statistics

Positions are metres from the top-left corner (x along the pitch, y down): the challenge ATD's 0-1
coordinates times the pitch size. tracking.npz keeps int32 millimetres (ball_mm, slots_mm;
-2147483648 = missing; v1 and v2 kept int16 centimetres), truth.npz int16 centimetres (-32768 =
missing). Load with load_match(), which reads both.
"""

import argparse
import bisect
import csv
import json
import math
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

FPS = 25
SLOTS_PER_TEAM = 23                    # 46 slots, as in the challenge ATD
PAD_BEFORE_S, HALFTIME_S, PAD_AFTER_S = 20, 60, 10
MISSING_CM = -32768
MISSING_MM = -2 ** 31

IDSSE = {
    "J03WMX": ("1. FC Köln vs. FC Bayern München", "Bundesliga"),
    "J03WN1": ("VfL Bochum 1848 vs. Bayer 04 Leverkusen", "Bundesliga"),
    "J03WPY": ("Fortuna Düsseldorf vs. 1. FC Nürnberg", "2. Bundesliga"),
    "J03WOH": ("Fortuna Düsseldorf vs. SSV Jahn Regensburg", "2. Bundesliga"),
    "J03WQQ": ("Fortuna Düsseldorf vs. FC St. Pauli", "2. Bundesliga"),
    "J03WOY": ("Fortuna Düsseldorf vs. F.C. Hansa Rostock", "2. Bundesliga"),
    "J03WR9": ("Fortuna Düsseldorf vs. 1. FC Kaiserslautern", "2. Bundesliga"),
}
FIGSHARE_IDS = {   # the file ids kloppy uses (kloppy._providers.sportec.get_IDSSE_url)
    "J03WPY": {"meta": 51643487, "tracking": 51643526},
    "J03WN1": {"meta": 51643472, "tracking": 51643517},
    "J03WMX": {"meta": 51643475, "tracking": 51643514},
    "J03WOH": {"meta": 51643478, "tracking": 51643520},
    "J03WQQ": {"meta": 51643484, "tracking": 51643529},
    "J03WOY": {"meta": 51643481, "tracking": 51643523},
    "J03WR9": {"meta": 51643490, "tracking": 51643532},
}
FIGSHARE_URL = "https://springernature.figshare.com/ndownloader/files/{}"
HF_TREE = "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main?recursive=true"
HF_FILE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"

METRICA = {"METRICA1": 1, "METRICA2": 2, "METRICA3": 3}
METRICA_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{g}/{name}"
METRICA_FILES = {
    1: ["Sample_Game_1_RawEventsData.csv", "Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["Sample_Game_2_RawEventsData.csv", "Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"],
}
METRICA_PITCH = (105.0, 68.0)          # stated in the sample-data README
MAX_DEAD_S = 60.0                      # no restart this long after a stop: the next event restarts play

# Split by MATCH, never by frame.
DEFAULT_SPLITS = {"train": ["J03WPY", "J03WOH", "J03WQQ", "J03WN1"],
                  "val": ["J03WOY", "J03WMX"],
                  "test": ["J03WR9"],
                  "val_metrica": ["METRICA1", "METRICA2"],
                  "test_metrica": ["METRICA3"]}

CITATION = {
    "idsse": ("Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). An integrated dataset of "
              "spatiotemporal and event data in elite soccer. Scientific Data, 12. Licensed CC BY 4.0."),
    "metrica": "Metrica Sports sample data, https://github.com/metrica-sports/sample-data",
    "fake": "synthetic match from tools/make_fake_data.py",
}

# Degradation. Players: 20.5 tracked objects (DATA.md). Ball: see degrade_ball and REAL_BALL_GAPS;
# the "calibrated" values are replaced by calibrate_ball and calibrate_tracking when the IDSSE train
# matches are built.
DEGRADE = {
    "fill_short_s": 5.0,          # shorter stretches without a ball are filled in a straight line (real ATD: none < 2 s)
    "occlusion_share": 0.175,     # NOT calibrated (nothing measures it on the real ATD): graded frames the tracker
                                  # briefly loses (crowds, air, speed); each training seed draws it from TRAIN_OCCLUSION
    "p_off": 1.0,                 # calibrated: probability that a stretch with the ball off the pitch hides it
    "p_dead": 0.6,                # calibrated: probability that a stoppage hides the ball
    "dead_start_s": (-0.5, 2.0),  # the hidden stretch starts this long after the stoppage (uniform)
    "dead_end_s": (-1.0, 1.0),    # and ends this long after the restart (uniform)
    "r_live": 0.5,                # calibrated: stretches without a ball in play, per live minute
    "live_min_s": 5.0,            # they last at least this long: shorter losses are occlusions, filled
    "live_median_s": 4.0,         # calibrated: median of their length beyond live_min_s (log-normal)
    "live_sigma": 0.7,            # calibrated: log-normal spread
    "fill_dead": 0.0,             # calibrated for Metrica: share of the stoppages without a ball in the
                                  # source that get an estimated position (IDSSE always has the ball)
    "ball_smooth_frames": 8.0,        # calibrated: sigma of the tracker's centred Gaussian smoother, ball
    "ball_error_d2_m": 0.006,         # calibrated: per-axis std of the second difference of the ball's position error
    "ball_error_m": 0.265,            # NOT calibrated (the second difference cannot tell 3 cm from 50 cm): RMS of
                                      # that error per axis; training seeds draw it from TRAIN_BALL_ERROR, this is
                                      # the middle of that range
    "player_detection_noise_m": 0.0,  # calibrated: white noise of the players' detections, before smoothing
    "player_smooth_frames": 2.0,      # calibrated
    "air_height_m": 1.8,
    "player_loss": 0.0025,
    "player_gap_s": 1.2,
    "swap_same_team": 0.02,
    "swap_cross_team": 0.004,
    "fragment": 0.0004,
    "duplicate": 0.0002,
}
# tools/ball_gap_diagnostic.py on the challenge ATD: aggregates without labels, run by a person. The
# only use of the challenge match in this file: the shape of its ball gaps, never its possession.
REAL_BALL_GAPS = {
    "missing": 0.411,
    "by_length": [0.000, 0.000, 0.001, 0.060, 0.168, 0.182],   # < 1, 1-2, 2-5, 5-10, 10-30, > 30 s
    "long_per_half": 38.5,                                      # stretches >= 10 s (77 in all)
    "set_pieces_preceded": 0.842, "set_pieces_chance": 0.638,  # 57 windows in the halves, median 35 s
    "where": [0.182, 0.091, 0.091, 0.636],                     # last seen before a stretch >= 10 s: off the
                                                                # pitch, < 2 m from a touchline / a goal line, inside
    # Not calibrated on (they depend on TAG_OFFSET, set from the diagnostic's offset histogram):
    "near_window_start": 0.105, "near_window": 0.513,          # stretch >= 10 s ending <= 3 s from a window start / window
    # (stretch >= 10 s, window) pairs by end - window start, 2.5 s bins from -10 s: observed and
    # expected with the windows shifted at random. TAG_OFFSET comes from their difference.
    "offset_observed": [3, 0, 0, 5, 1, 8, 12, 4, 3, 1, 0, 2, 1, 1, 1, 0, 1, 2],
    "offset_shifted": [1.6, 1.6, 1.7, 1.7, 1.6, 1.7, 1.3, 1.4, 1.6, 1.7, 1.7, 1.6, 1.5, 1.6, 1.6, 1.4, 1.7, 1.7],
    # |second difference| of the positions, metres per frame^2, graded frames (measure 8): how smooth the
    # tracker is. The players' within each tracklet.
    "ball_d2": {"p10": 0.0014, "p50": 0.0091, "p90": 0.0287, "p95": 0.0353, "p99": 0.0510, "p99.9": 0.0961},
    "player_d2": {"p10": 0.0007, "p50": 0.0014, "p90": 0.0043, "p99": 0.0096},
    # SET PIECES windows per Type in the whole file, and the graded minutes from the two kick-offs.
    "set_pieces_by_type": {"Throw-in": 26, "Corner Kick": 12, "Goal Kick": 10, "Free Kick": 5,
                           "Kick Off (After Goal)": 4, "Kick Off (Start)": 2},
    "graded_minutes": 103.9,
}
BALL_GRID = {"p_off": (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
             "p_dead": (0.2, 0.4, 0.6, 0.8, 1.0),
             "r_live": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
             "live_median_s": (0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
             "live_sigma": (0.25, 0.4, 0.7, 1.0)}
FILL_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CALIBRATION_DRAWS = 3          # degradations per match pooled for each setting
# The model tracker, scored on the real ATD's second-difference percentiles (REAL_BALL_GAPS ball_d2 /
# player_d2). p10 is left out: ~0 everywhere, set by the millimetre rounding.
TRACK_GRID_BALL = {"ball_smooth_frames": (6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
                   "ball_error_d2_m": (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.013, 0.016, 0.02)}
TRACK_GRID_PLAYER = {"player_smooth_frames": (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
                     "player_detection_noise_m": (0.0, 0.01, 0.02, 0.05)}
BALL_D2_FIT = (50, 90, 95, 99, 99.9)
PLAYER_D2_FIT = (50, 90, 99)
# Training seeds: the calibrated parameters each scaled by a factor in these ranges, and an occlusion
# share drawn from TRAIN_OCCLUSION. The calibration draws them too, so train reproduces the target on average.
TRAIN_JITTER = {"p_off": (0.85, 1.15), "p_dead": (0.85, 1.15), "r_live": (0.75, 1.33), "live_median_s": (0.8, 1.25)}
TRAIN_OCCLUSION = (0.05, 0.3)
TRAIN_BALL_ERROR = (0.03, 0.5)          # ball_error_m of each training seed (uniform), not calibrated
# Smart Tagging: SET PIECES windows of the real file last 35 s, kick-offs 5 s. Where the restart falls
# inside is a distribution: "uniform" (lo_s, hi_s) or "histogram" (edges_s, weights; uniform within a
# bin). TAG_OFFSET is the excess of the diagnostic's offset histogram on the real file over the
# shifted-window background (REAL_BALL_GAPS offset_*), in 5 s bins: all the excess lies between
# -2.5 and 12.5 s, 2.5 s bins hold 1-12 pairs, and the gap end marks the restart within ~1 s.
TAG_WINDOW_S = 35.0
TAG_OFFSET = {"kind": "histogram", "edges_s": [-2.5, 2.5, 7.5, 12.5], "weights": [2.7, 17.0, 4.0]}
TAG_KICKOFF_WINDOW_S = 5.0
TAG_KICKOFF_OFFSET = {"kind": "uniform", "lo_s": 2.0, "hi_s": 3.0}
TAG_OFFSET_N = 16              # training seeds draw the TAG_OFFSET weights from a Dirichlet worth this many
                               # observations (the real histogram's excess holds ~16-24)
# Smart Tagging recall per restart type: the real file's windows per 90 graded minutes (REAL_BALL_GAPS
# set_pieces_by_type) over the restarts the sources have. Free kicks: ~30 per 90 min in the sources,
# 4.3 tagged in the real file, about 1 in 7. Corners: the sources have 8 per 90 min, the real file 10.4, so all.
TAG_RECALL = {"Throw-in": 0.55, "Corner Kick": 1.0, "Goal Kick": 0.55, "Free Kick": 0.14,
              "Kick Off (Start)": 1.0, "Kick Off (After Goal)": 1.0}
CORNER_RADIUS_M = 5.0          # without event types, a restart this close to a corner flag is a corner: the
                               # arc is 1 m, plus tracking error and a ball already moving at the restart frame
TAG_WRONG_TEAM = 0.05
TAG_FALSE = 0.03

# Label harmonisation: fixed artefact clean-ups for every source, plus the control rule for IDSSE.
MIN_DEAD_S = 1.0                       # a real stoppage lasts seconds
MIN_RESTART_SPELL_S = 0.2              # a flicker right after a restart is an artefact
CONTROL_M, CONTROL_S = 1.5, 0.4        # team choice; the grid and cross-check are in label_harmonization.json
CONTROL_CHOICE = (
    "The grid settings along the diagonal (1.0 m / 0.2 s, 1.5 m / 0.4 s, 2.0 m / 0.5-0.6 s) are equivalent: their "
    "losses differ by less than the loss between the two reference games themselves. 1.5 m / 0.4 s is chosen on "
    "physical grounds: 1.5 m is a player's reach plus the tracking error of player and ball positions, and 0.4 s is "
    "about the time needed to control the ball and play it.")
CONTROL_GAP = 3                        # frames: a ball-tracking blip does not break a control spell
CONTROL_MARGIN = 5                     # frames either side of a spell also searched for control
CONTROL_GRID_M = (0.75, 1.0, 1.5, 2.0)
CONTROL_GRID_S = (0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


# ==================================================================================================
# Download
# ==================================================================================================

def _download(url, path):
    tmp = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "mtah-build/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
        # figshare's WAF answers non-browser clients with "202 Accepted" and an empty body
        if response.status != 200:
            raise IOError("HTTP %d from %s" % (response.status, url))
        total, done = int(response.headers.get("Content-Length") or 0), 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print("\r    %s  %5.1f / %5.1f MB" % (os.path.basename(path), done / 1e6, total / 1e6), end="")
    print()
    if done == 0:
        os.remove(tmp)
        raise IOError("empty response from %s" % url)
    os.replace(tmp, path)


def fetch_idsse(match, raw_dir):
    """(meta_path, positions_path), downloading whatever is missing."""
    os.makedirs(raw_dir, exist_ok=True)
    paths = {"meta": os.path.join(raw_dir, "%s_meta.xml" % match),
             "tracking": os.path.join(raw_dir, "%s_positions.xml" % match)}
    for kind, path in paths.items():
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            continue
        try:
            _download(FIGSHARE_URL.format(FIGSHARE_IDS[match][kind]), path)
            continue
        except Exception as error:                                    # noqa: BLE001
            print("    figshare failed (%s); trying the Hugging Face mirror" % error)
        _download(HF_FILE.format(_hf_name(match, kind)), path)
    return paths["meta"], paths["tracking"]


_HF_LIST = None


def _hf_name(match, kind):
    global _HF_LIST
    if _HF_LIST is None:
        with urllib.request.urlopen(HF_TREE, timeout=60) as response:
            _HF_LIST = [item["path"] for item in json.load(response) if item.get("type") == "file"]
    words = {"meta": ("matchinformation", "meta"), "tracking": ("position", "tracking")}[kind]
    hits = [p for p in _HF_LIST if match in p and any(w in p.lower() for w in words)]
    if len(hits) != 1:
        raise IOError("cannot identify the %s file of %s on Hugging Face: %s" % (kind, match, hits))
    return hits[0]


def fetch_metrica(match, raw_dir):
    """Folder with the sample game's files, downloading whatever is missing (GitHub)."""
    g = METRICA[match]
    folder = os.path.join(raw_dir, "Sample_Game_%d" % g)
    os.makedirs(folder, exist_ok=True)
    for name in METRICA_FILES[g]:
        path = os.path.join(folder, name)
        if not (os.path.isfile(path) and os.path.getsize(path) > 1000):
            _download(METRICA_URL.format(g=g, name=name), path)
    return folder


# ==================================================================================================
# Parse IDSSE (DFL XML)
# ==================================================================================================

def timeline(period_ranges):
    """Original frame ranges per period -> (offsets, halves, n) on our padded timeline."""
    offsets, halves, pos = [], [], PAD_BEFORE_S * FPS
    for i, (first, last) in enumerate(period_ranges):
        offsets.append(pos - first)
        halves.append((i + 1, pos, pos + last - first))
        pos += last - first + 1 + (HALFTIME_S * FPS if i == 0 else 0)
    return offsets, halves, pos + PAD_AFTER_S * FPS


def parse_dfl(meta_path, positions_path, match):
    """The clean match: players and ball in metres from the top-left corner, y pointing down."""
    meta = ET.parse(meta_path).getroot()
    teams = {t.get("TeamId"): (t.get("Role"), t.get("TeamName")) for t in meta.iter("Team")}
    env = next(meta.iter("Environment"), None)
    pitch = (float(env.get("PitchX")), float(env.get("PitchY"))) if env is not None else None

    sets = {}
    cur, buf = None, None
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start":
            if elem.tag == "FrameSet":
                cur = (elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId"))
                buf = {"N": [], "X": [], "Y": [], "Z": [], "P": [], "S": []}
            elif elem.tag == "PitchSize" and pitch is None:
                pitch = (float(elem.get("X")), float(elem.get("Y")))
            continue
        if elem.tag == "Frame" and buf is not None:
            a = elem.attrib
            buf["N"].append(int(a["N"]))
            buf["X"].append(float(a["X"]))
            buf["Y"].append(float(a["Y"]))
            if cur[1] == "BALL":
                buf["Z"].append(float(a.get("Z", "nan")))
                buf["P"].append(int(a.get("BallPossession", "0") or 0))
                buf["S"].append(int(a.get("BallStatus", "-1") or -1))
            elem.clear()
        elif elem.tag == "FrameSet":
            sets[cur] = {k: np.asarray(v) for k, v in buf.items() if len(v)}
            cur, buf = None, None
            elem.clear()
    if pitch is None:
        raise ValueError("no pitch size in %s or %s" % (meta_path, positions_path))
    L, W = pitch

    sections = [s for s in ("firstHalf", "secondHalf") if any(k[0] == s and k[1] == "BALL" for k in sets)]
    ranges = []
    for s in sections:
        ball_set = [v for k, v in sets.items() if k[0] == s and k[1] == "BALL"][0]
        ranges.append((int(ball_set["N"].min()), int(ball_set["N"].max())))
    offs, halves, n = timeline(ranges)
    offsets = dict(zip(sections, offs))

    people = sorted({(k[1], k[2]) for k in sets if k[1] != "BALL"})
    index = {person: i for i, person in enumerate(people)}
    player_team = np.array([0 if teams.get(t, ("?",))[0] == "home" else 1 for t, _ in people], dtype=np.int8)
    players = np.full((n, len(people), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    status = np.full(n, -1, dtype=np.int8)
    possession = np.full(n, -1, dtype=np.int8)
    for (section, team, person), v in sets.items():
        if section not in offsets:
            continue
        rows = v["N"] + offsets[section]
        x, y = v["X"] + L / 2.0, W / 2.0 - v["Y"]                 # centre origin, y up -> top-left, y down
        if team == "BALL":
            ball[rows, 0], ball[rows, 1] = x, y
            if "Z" in v:
                ball[rows, 2] = v["Z"]
            status[rows] = v["S"]
            possession[rows] = np.where(v["P"] == 1, 0, np.where(v["P"] == 2, 1, -1))
        else:
            k = index[(team, person)]
            players[rows, k, 0], players[rows, k, 1] = x, y

    home_name = next((name for role, name in teams.values() if role == "home"), "home")
    away_name = next((name for role, name in teams.values() if role != "home"), "away")
    title, division = IDSSE.get(match, ("%s vs %s" % (home_name, away_name), "?"))
    clean = make_clean(match, title, division, home_name, away_name, (L, W), halves,
                       players, player_team, ball, status, possession)
    clean.update(source="idsse", label_rule="D where BallStatus=0, else the BallPossession team (DFL)")
    return clean


# ==================================================================================================
# Parse Metrica sample games (tracking + events -> labels)
# ==================================================================================================

def labels_from_events(ev, n_frames):
    """Per original frame, H/A/D from Metrica's human event data, following its event definitions.

      stop    -> D : BALL OUT (end), SHOT out or goal (end), FAULT RECEIVED (start), a CHALLENGE
                     whose subtype has FAULT but not ADVANTAGE (start; many fouls are only tagged
                     this way), BALL LOST or PASS that is forced / end of half / offside / hand ball (end)
      restart      : SET PIECE, or a GOAL KICK pass (start) -> that team, ball alive
      possession   : PASS, RECOVERY, SHOT, CARRY (start) -> that team, while the ball is alive
    While the ball is dead only a restart brings it back: the events in between -- the keeper
    fetching the ball after a shot wide, play the referee has already stopped for offside -- do
    not. Two exceptions, where the stoppage has no tagged restart of its own, and the first
    possession event restarts play: no restart follows within MAX_DEAD_S of the stop (a dropped
    ball), or another stop comes before the next restart (that restart belongs to the later stop;
    the tagger missed a quick restart). CHALLENGE, BALL LOST and CARD do not change possession by themselves:
    the opponent's RECOVERY or PASS that follows does, at its start frame.
    """
    points = []
    for e in ev:
        sub = e["sub"].upper()
        if e["type"] == "SET PIECE" or ("GOAL KICK" in sub and e["type"] in ("PASS", "BALL LOST", "BALL OUT")):
            points.append((e["start"], 0, "restart", e["team"]))
        elif e["type"] in ("PASS", "RECOVERY", "SHOT", "CARRY"):
            points.append((e["start"], 1, "own", e["team"]))
        if e["type"] == "BALL OUT" or (e["type"] == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
            points.append((max(e["end"], e["start"]), 2, "stop", None))
        elif e["type"] == "FAULT RECEIVED" or (e["type"] == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub):
            points.append((e["start"], 2, "stop", None))
        elif e["type"] in ("BALL LOST", "PASS") and any(k in sub for k in ("FORCED", "END HALF", "OFFSIDE", "HAND BALL")):
            points.append((max(e["end"], e["start"]), 2, "stop", None))
    points.sort(key=lambda p: (p[0], p[1]))
    restarts = [f for f, _, kind, _ in points if kind == "restart"]
    stops = [f for f, _, kind, _ in points if kind == "stop"]
    lab = np.full(n_frames + 1, "D", dtype="<U1")
    state, team, last, stopped = "D", None, 0, 0
    for f, _, kind, t in points:
        f = min(max(f, 0), n_frames)
        lab[last:f] = "D" if state == "D" else team
        if kind == "restart":
            state, team = "live", t
        elif kind == "own":
            if state == "live":
                team = t
            else:
                i = bisect.bisect_right(restarts, stopped)          # first restart after the stop
                nxt = restarts[i] if i < len(restarts) else None
                j = bisect.bisect_right(stops, f)                    # first stop after this event
                if nxt is None or nxt > stopped + MAX_DEAD_S * FPS or (j < len(stops) and stops[j] < nxt):
                    state, team = "live", t
        else:
            if state == "live":
                stopped = f
            state = "D"
        last = f
    lab[last:] = "D"
    return lab


def metrica_restart_kind(e):
    """The Smart Tagging Type of a Metrica restart event ("Kick Off" for either kick-off), or None."""
    sub = e["sub"].upper()
    if e["type"] == "SET PIECE":
        for key, kind in (("THROW", "Throw-in"), ("CORNER", "Corner Kick"), ("KICK OFF", "Kick Off"),
                          ("FREE KICK", "Free Kick"), ("PENALTY", "Free Kick")):
            if key in sub:
                return kind
        return None
    if "GOAL KICK" in sub and e["type"] in ("PASS", "BALL LOST", "BALL OUT"):
        return "Goal Kick"
    return None


def parse_metrica(match, folder):
    import pandas as pd
    g = METRICA[match]
    L, W = METRICA_PITCH
    if g in (1, 2):
        base = os.path.join(folder, "Sample_Game_%d_" % g)
        tables, players_cols = [], []
        for t, side in ((0, "Home"), (1, "Away")):
            df = pd.read_csv(base + "RawTrackingData_%s_Team.csv" % side, skiprows=2)
            tables.append(df)
            cols = list(df.columns)
            players_cols += [(t, df.iloc[:, i].to_numpy(np.float32), df.iloc[:, i + 1].to_numpy(np.float32))
                             for i, c in enumerate(cols) if c.startswith("Player")]
        home = tables[0]
        frames = home["Frame"].to_numpy(int)
        period = home["Period"].to_numpy(int)
        b = list(home.columns).index("Ball")
        bx, by = home.iloc[:, b].to_numpy(np.float32), home.iloc[:, b + 1].to_numpy(np.float32)
        ranges = [(int(frames[period == p].min()), int(frames[period == p].max())) for p in (1, 2)]
        evdf = pd.read_csv(base + "RawEventsData.csv")
        events = [{"team": "H" if team == "Home" else "A", "type": kind, "sub": "" if pd.isna(sub) else str(sub),
                   "start": int(start), "end": int(end)}
                  for team, kind, sub, start, end in zip(evdf["Team"], evdf["Type"], evdf["Subtype"],
                                                         evdf["Start Frame"], evdf["End Frame"])]
        home_name, away_name = "Home", "Away"
        n_orig = int(frames.max())
        raw_players = [(t, frames, x, y) for t, x, y in players_cols]
        raw_ball = (frames, bx, by)
    else:
        meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
        pc = {e.get("id"): e.get("playerId") for e in meta.iter("PlayerChannel")}
        teams = {t.get("id"): t.findtext("Name") for t in meta.iter("Team")}
        score = next(meta.iter("Score"), None)
        home_id = score.get("idLocalTeam") if score is not None else list(teams)[0]
        team_of = {p.get("id"): (0 if p.get("teamId") == home_id else 1) for p in meta.iter("Player")}
        params = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")}
        ranges = [(int(params["first_half_start"]), int(params["first_half_end"])),
                  (int(params["second_half_start"]), int(params["second_half_end"]))]
        specs = []
        for s in meta.iter("DataFormatSpecification"):
            order = [pc[r.get("playerChannelId")] for r in s.iter("PlayerChannelRef")
                     if r.get("playerChannelId").endswith("_x")]
            specs.append((int(s.get("startFrame")), int(s.get("endFrame")), order))
        ids = sorted(team_of)
        col = {pid: i for i, pid in enumerate(ids)}
        n_orig = ranges[-1][1]
        px = np.full((n_orig + 1, len(ids)), np.nan, dtype=np.float32)
        py = np.full((n_orig + 1, len(ids)), np.nan, dtype=np.float32)
        bx = np.full(n_orig + 1, np.nan, dtype=np.float32)
        by = np.full(n_orig + 1, np.nan, dtype=np.float32)
        spec_i = 0
        with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
            for line in handle:
                head, body, ball_txt = line.rstrip("\r\n").split(":")
                f = int(head)
                while specs[spec_i][1] < f:
                    spec_i += 1
                order = specs[spec_i][2]
                for k, grp in enumerate(body.split(";")):
                    x, y = grp.split(",")[:2]
                    if x != "NaN":
                        c = col[order[k]]
                        px[f, c], py[f, c] = float(x), float(y)
                x, y = ball_txt.split(",")[:2]
                if x != "NaN":
                    bx[f], by[f] = float(x), float(y)
        frames = np.arange(n_orig + 1)
        raw_players = [(team_of[pid], frames, px[:, col[pid]], py[:, col[pid]]) for pid in ids]
        raw_ball = (frames, bx, by)
        with open(os.path.join(folder, "Sample_Game_3_events.json"), encoding="utf-8") as handle:
            data = json.load(handle)["data"]
        events = []
        for e in data:
            subs = e.get("subtypes")
            subs = [subs] if isinstance(subs, dict) else (subs or [])
            events.append({"team": "H" if e["team"]["id"] == home_id else "A", "type": e["type"]["name"],
                           "sub": "-".join(s["name"] for s in subs), "start": int(e["start"]["frame"]),
                           "end": int(e["end"]["frame"] if e["end"]["frame"] is not None else e["start"]["frame"])})
        home_name, away_name = teams[home_id], [v for k, v in teams.items() if k != home_id][0]

    lab_orig = labels_from_events(events, n_orig)
    offsets, halves, n = timeline(ranges)
    restart_kinds = []                                  # (frame on the new timeline, Type) from the events
    for e in events:
        kind = metrica_restart_kind(e)
        for (first, last), off in zip(ranges, offsets):
            if kind and first <= e["start"] <= last:
                restart_kinds.append((int(e["start"] + off), kind))
    P = len(raw_players)
    players = np.full((n, P, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    status = np.full(n, -1, dtype=np.int8)
    possession = np.full(n, -1, dtype=np.int8)
    for (first, last), off in zip(ranges, offsets):
        orig = np.arange(first, last + 1)
        rows = orig + off
        lab = lab_orig[orig]
        status[rows] = np.where(lab == "D", 0, 1)
        possession[rows] = np.where(lab == "H", 0, np.where(lab == "A", 1, -1))
        f_ball, x, y = raw_ball
        sel = (f_ball >= first) & (f_ball <= last)
        ball[f_ball[sel] + off, 0], ball[f_ball[sel] + off, 1] = x[sel] * L, y[sel] * W
        for k, (_, f_pl, x, y) in enumerate(raw_players):
            sel = (f_pl >= first) & (f_pl <= last)
            players[f_pl[sel] + off, k, 0], players[f_pl[sel] + off, k, 1] = x[sel] * L, y[sel] * W
    player_team = np.array([t for t, *_ in raw_players], dtype=np.int8)
    clean = make_clean(match, "Metrica Sports sample game %d (anonymised)" % g, "?", home_name, away_name,
                       (L, W), halves, players, player_team, ball, status, possession)
    clean.update(source="metrica", label_rule="derived from Metrica's human event data (labels_from_events)",
                 restart_kinds=sorted(restart_kinds))
    return clean


# ==================================================================================================
# The clean match, labels and harmonisation
# ==================================================================================================

def make_clean(match, title, division, home, away, pitch, halves, players, player_team, ball, status, possession):
    """Labels from status/possession, and sides that swap at half-time like a real camera."""
    L, W = pitch
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        graded[a:b + 1] = True

    flipped = False
    if len(halves) == 2:
        means = []
        for _, a, b in halves:
            with np.errstate(invalid="ignore"), _quiet():
                means.append(np.nanmean(players[a:b + 1][:, player_team == 0, 0]) - L / 2.0)
        if np.sign(means[0]) == np.sign(means[1]):
            _, a, b = halves[1]
            players[a:b + 1, :, 0] = L - players[a:b + 1, :, 0]
            players[a:b + 1, :, 1] = W - players[a:b + 1, :, 1]
            ball[a:b + 1, 0] = L - ball[a:b + 1, 0]
            ball[a:b + 1, 1] = W - ball[a:b + 1, 1]
            flipped = True

    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        last = -1
        for f in range(a, b + 1):
            if possession[f] >= 0:
                last = possession[f]
            if status[f] == 0:
                labels[f] = "D"
            elif status[f] == 1 and last >= 0:
                labels[f] = "HA"[last]
    return {"match": match, "title": title, "division": division, "home": home, "away": away,
            "pitch": (float(L), float(W)), "halves": [tuple(int(v) for v in h) for h in halves], "n": int(n),
            "players": players, "player_team": player_team, "ball": ball, "status": status,
            "possession": possession, "labels": labels, "labels_raw": labels.copy(), "graded": graded,
            "flipped_second_half": flipped}


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


def _runs(seg):
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate(([0], cut, [len(seg)]))
    return [[int(p), int(q), seg[p]] for p, q in zip(bounds[:-1], bounds[1:])]


def clean_artefacts(labels, halves):
    """Fixed clean-ups, every source: stoppages < MIN_DEAD_S inside a half are not stoppages (the
    previous team keeps the ball), and a team spell < MIN_RESTART_SPELL_S right after a stoppage
    is a flicker (the next team had it from the restart)."""
    out = labels.copy()
    for _, a, b in halves:
        seg = out[a:b + 1]
        runs = _runs(seg)
        for i in range(1, len(runs) - 1):
            p, q, v = runs[i]
            if v == "D" and q - p < MIN_DEAD_S * FPS and runs[i - 1][2] in ("H", "A") and runs[i + 1][2] in ("H", "A"):
                seg[p:q] = runs[i - 1][2]
        runs = _runs(seg)
        for i in range(1, len(runs) - 1):
            p, q, v = runs[i]
            if v in ("H", "A") and runs[i - 1][2] == "D" and runs[i + 1][2] in ("H", "A") \
                    and q - p < MIN_RESTART_SPELL_S * FPS:
                seg[p:q] = runs[i + 1][2]
    return out


def ball_distances(clean):
    """(distance of every player to the ball [n, P], inf where either is missing; player teams)."""
    players, ball = clean["players"], clean["ball"][:, :2]
    dist = np.full(players.shape[:2], np.inf, dtype=np.float32)
    for s in range(0, len(ball), 20000):
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(players[s:s + 20000] - ball[s:s + 20000, None, :], axis=2)
        dist[s:s + 20000] = np.where(np.isnan(d), np.inf, d)
    return dist, clean["player_team"].astype(int)


def team_control(dist, team, control_m, control_s):
    """[n, 2] bool: a player of that team keeps the ball within control_m for >= control_s
    (clean tracking; gaps of up to CONTROL_GAP frames do not break a spell)."""
    ctrl = np.zeros((len(dist), 2), dtype=bool)
    for k in range(dist.shape[1]):
        near = dist[:, k] <= control_m
        for p, q, v in _runs(near):
            if not v and 0 < p and q < len(near) and q - p <= CONTROL_GAP:
                near[p:q] = True
        for p, q, v in _runs(near):
            if v and q - p >= control_s * FPS:
                ctrl[p:q, team[k]] = True
    return ctrl


def merge_uncontrolled(labels, halves, ctrl):
    """CHALLENGE.md: a touch, a deflection or a lost aerial duel is not a change of possession.
    Inside each live stretch, a team spell whose neighbours both belong to the other team -- or
    that ends the stretch right after the other team's spell -- goes back to that team if no player
    of its own team controlled the ball during it (ctrl, from team_control). The first spell after
    a restart is never touched: "there is always a clear possessor". Shortest spells first."""
    out = labels.copy()
    for _, a, b in halves:
        seg = out[a:b + 1]
        for p0, q0, v0 in _runs(seg == "D"):
            if v0:
                continue
            runs = _runs(seg[p0:q0])
            while len(runs) > 1:
                best = None
                for i in range(1, len(runs)):
                    p, q, v = runs[i]
                    if v not in ("H", "A") or not (i == len(runs) - 1 or runs[i + 1][2] == runs[i - 1][2]):
                        continue
                    lo, hi = a + p0 + p - CONTROL_MARGIN, a + p0 + q + CONTROL_MARGIN
                    if ctrl[max(lo, 0):hi, "HA".index(v)].any():
                        continue
                    if best is None or q - p < runs[best][1] - runs[best][0]:
                        best = i
                if best is None:
                    break
                p, q, _ = runs[best]
                seg[p0 + p:p0 + q] = runs[best - 1][2]
                runs = _runs(seg[p0:q0])
    return out


def possession_stats(labels, halves):
    lengths, per_half = [], []
    for _, a, b in halves:
        runs = _runs(labels[a:b + 1])
        team_runs = [q - p for p, q, v in runs if v in ("H", "A")]
        per_half.append(len(team_runs))
        lengths += team_runs
    g = np.concatenate([labels[a:b + 1] for _, a, b in halves])
    minutes = len(g) / FPS / 60.0
    alive_min = np.isin(g, ("H", "A")).sum() / FPS / 60.0
    secs = np.array(lengths) / FPS if lengths else np.zeros(1)
    return {"possessions": int(sum(per_half)), "per_half": per_half,
            "mean_possession_s": round(float(secs.mean()), 2), "median_possession_s": round(float(np.median(secs)), 2),
            "share_under_1s": round(float(np.mean(secs < 1.0)), 4),
            "possessions_per_min": round(sum(per_half) / minutes, 2),
            "possessions_per_alive_min": round(sum(per_half) / alive_min, 2) if alive_min else 0.0,
            "share": {k: round(float(np.mean(g == k)), 4) for k in ("H", "A", "D")},
            "graded_minutes": round(minutes, 1)}


def _durations(cleans):
    out = []
    for c in cleans:
        for _, a, b in c["halves"]:
            out += [q - p for p, q, v in _runs(c["labels"][a:b + 1]) if v in ("H", "A")]
    return np.sort(np.array(out, dtype=float) / FPS)


def distribution_loss(cleans, reference):
    """How far the possession durations of `cleans` are from the reference's: the KS distance
    between the two duration distributions plus the relative error in possessions per live minute."""
    x, y = _durations(cleans), _durations(reference)
    grid = np.union1d(x, y)
    ks = float(np.max(np.abs(np.searchsorted(x, grid, "right") / len(x) - np.searchsorted(y, grid, "right") / len(y))))
    def rate(cs):
        return float(np.mean([possession_stats(c["labels"], c["halves"])["possessions_per_alive_min"] for c in cs]))
    r, rr = rate(cleans), rate(reference)
    return {"ks": round(ks, 4), "rate": round(r, 2), "rate_reference": round(rr, 2),
            "median_s": round(float(np.median(x)), 2), "under_1s": round(float(np.mean(x < 1)), 4),
            "under_2s": round(float(np.mean(x < 2)), 4), "under_5s": round(float(np.mean(x < 5)), 4),
            "loss": round(ks + abs(r - rr) / rr, 4)}


def calibrate_control(train, refs, chosen):
    """The control rule over a grid, scored against the reference games' duration distribution
    (all of them, and each one alone), plus a cross-check: fit on one game, score on the other."""
    dists = [ball_distances(c) for c in train]
    grid = []
    for m in CONTROL_GRID_M:
        for s in CONTROL_GRID_S:
            merged = [dict(c, labels=merge_uncontrolled(c["labels"], c["halves"], team_control(d, t, m, s)))
                      for c, (d, t) in zip(train, dists)]
            row = {"control_m": m, "control_s": s, "all": distribution_loss(merged, list(refs.values()))}
            row.update({name: distribution_loss(merged, [ref])["loss"] for name, ref in refs.items()})
            grid.append(row)
    best = min(grid, key=lambda r: r["all"]["loss"])
    cross = []
    names = sorted(refs)
    for fit in names:
        for check in names:
            if check != fit:
                pick = min(grid, key=lambda r: r[fit])
                cross.append({"fit_on": fit, "picks": [pick["control_m"], pick["control_s"]],
                              "loss_on_fit": pick[fit], "check_on": check, "loss_on_check": pick[check],
                              "game_to_game_loss": distribution_loss([refs[fit]], [refs[check]])["loss"]})
    picked = next(r for r in grid if [r["control_m"], r["control_s"]] == list(chosen))
    unmerged = distribution_loss(train, list(refs.values()))
    return {"references": names, "loss": "KS(durations) + |rate - rate_ref| / rate_ref", "grid": grid,
            "best_on_all": [best["control_m"], best["control_s"]], "chosen": list(chosen), "chosen_scores": picked,
            "no_merge": unmerged, "cross_validation": cross}


# ==================================================================================================
# Degrade into ATD-like tracking
# ==================================================================================================

def smooth_noise(rng, n, window):
    out = np.convolve(rng.normal(size=n + window), np.ones(window) / window, mode="valid")[:n]
    return (out - out.mean()) / out.std()


def rotate(vec, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def smooth_track(xy, sigma):
    """Centred Gaussian smoothing along time (axis 0) of positions with NaN where missing: each frame
    with a position becomes the weighted mean of the positions around it, missing frames left out.
    Centred, so no delay: the ATD is an offline product (an assumption)."""
    xy = np.asarray(xy, dtype=np.float64)
    if sigma <= 0:
        return xy.copy()
    from scipy.ndimage import gaussian_filter1d
    seen = np.isfinite(xy)
    num = gaussian_filter1d(np.where(seen, xy, 0.0), sigma, axis=0, mode="constant")
    den = gaussian_filter1d(seen.astype(np.float64), sigma, axis=0, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~seen] = np.nan
    return out


def _true_runs(mask):
    """[(start, end)] half-open runs of True."""
    m = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def ball_context(clean):
    """What degrade_ball needs from a match, computed once (calibration reuses it many times)."""
    ball, labels, n = clean["ball"], clean["labels"], clean["n"]
    L, W = clean["pitch"]
    present = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        off = present & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))
    dead, live = [], []
    for _, a, b in clean["halves"]:
        seg = labels[a:b + 1]
        dead += [(a + p, a + q) for p, q in _true_runs(seg == "D")]
        live.append(a + np.flatnonzero(np.isin(seg, ("H", "A"))))
    return {"n": n, "present": present, "off": off, "off_runs": _true_runs(off), "dead": dead, "live": live,
            "graded": clean["graded"]}


def occlusions(clean, rng, share, max_s=DEGRADE["fill_short_s"]):
    """Frames a tracker loses for a moment -- crowds, air, speed -- as a share of graded frames. A
    moment: each run is cut below max_s, so an occlusion alone is always filled (degrade_ball)."""
    ball, players, graded = clean["ball"].astype(np.float64), clean["players"], clean["graded"]
    n = len(ball)
    present = ~np.isnan(ball[:, 0])
    crowd = np.zeros(n)
    for s in range(0, n, 20000):
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(players[s:s + 20000] - ball[s:s + 20000, None, :2], axis=2)
            crowd[s:s + 20000] = np.nansum(d < 4.0, axis=1)
    speed = np.zeros(n)
    speed[1:] = np.nan_to_num(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * FPS)
    air = np.nan_to_num(ball[:, 2]) > DEGRADE["air_height_m"]
    score = 0.45 * crowd + 1.4 * air + 0.04 * np.minimum(speed, 30) + smooth_noise(rng, n, 21)
    if share <= 0 or not np.any(graded & present):
        return np.zeros(n, dtype=bool)
    lost = present & (score > np.quantile(score[graded & present], 1.0 - share))
    keep = int(max_s * FPS) - 1
    for p, q in _true_runs(lost):
        lost[p + keep:q] = False
    return lost


def hidden_ball(ctx, rng, prm, occluded):
    """Frames without a ball before short stretches are filled: missing in the source, off the
    pitch (each stretch out with probability p_off), hidden stoppages (each with probability
    p_dead), stretches in play of at least live_min_s (r_live per live minute, log-normal beyond
    live_min_s) and brief occlusions. Every stretch draws its random numbers whether it is hidden or not, so the
    settings compared by the calibration see the same draws."""
    n = ctx["n"]
    missing = ~ctx["present"] | occluded
    for (p, q), u in zip(ctx["off_runs"], rng.random(len(ctx["off_runs"]))):
        if u < prm["p_off"]:
            missing[p:q] = True
    dead = ctx["dead"]
    hide = rng.random(len(dead))
    starts = rng.uniform(*prm["dead_start_s"], len(dead))
    ends = rng.uniform(*prm["dead_end_s"], len(dead))
    for (p, q), u, ds, de in zip(dead, hide, starts, ends):
        if u < prm["p_dead"]:
            a, b = p + int(ds * FPS), q + int(de * FPS)
            missing[max(a, 0):max(min(b, n), 0)] = True
    for live in ctx["live"]:
        k = rng.poisson(prm["r_live"] * len(live) / FPS / 60.0) if len(live) else 0
        if k:
            starts = rng.choice(live, size=k)
            lengths = np.minimum(prm["live_min_s"] + rng.lognormal(math.log(prm["live_median_s"]), prm["live_sigma"], k),
                                 120.0)
            for f, d in zip(starts, lengths):
                missing[f:min(f + int(d * FPS), n)] = True
    return missing


def source_gaps(clean):
    """[(start, end)] stretches inside a half where the source itself has no ball, mostly during a
    stoppage, with a sighting on both sides."""
    present = ~np.isnan(clean["ball"][:, 0])
    dead = clean["labels"] == "D"
    n = len(present)
    return [(int(p), int(q)) for p, q in _true_runs(~present & clean["graded"])
            if 0 < p and q < n and present[p - 1] and present[q] and dead[p:q].mean() >= 0.5]


def fill_source_gaps(clean, rng, share, gaps=None):
    """The clean ball with a share of the source's stoppage gaps (source_gaps) given an estimated
    position: the last sighting carried in a straight line to the restart point (the next
    sighting). Metrica's corrected tracking drops the ball in most stoppages, more often than the
    ATD loses it there. Every gap draws its random number, as in hidden_ball."""
    ball = clean["ball"].copy()
    gaps = source_gaps(clean) if gaps is None else gaps
    for (p, q), u in zip(gaps, rng.random(len(gaps))):
        if u < share:
            t = ((np.arange(p, q) - (p - 1)) / (q - (p - 1)))[:, None]
            ball[p:q, :2] = ball[p - 1, :2] + t * (ball[q, :2] - ball[p - 1, :2])
    return ball


def short_stretches(missing, fill_s):
    """Stretches without a ball shorter than fill_s with a sighting on both sides."""
    out = np.zeros(len(missing), dtype=bool)
    for p, q in _true_runs(missing):
        if p > 0 and q < len(missing) and q - p < fill_s * FPS:
            out[p:q] = True
    return out


def fill_straight(xy, missing, fill_s):
    """xy [n, 2] with the stretches of `missing` shorter than fill_s (a sighting on both sides) filled
    in a straight line between those sightings and the longer ones NaN; returns (xy, still missing)."""
    fill = short_stretches(missing, fill_s)
    out = np.array(xy, dtype=np.float64)
    seen = np.flatnonzero(~missing)
    if fill.any() and len(seen):
        idx = np.flatnonzero(fill)
        last = seen[np.searchsorted(seen, idx) - 1]
        nxt = seen[np.searchsorted(seen, idx)]
        t = ((idx - last) / (nxt - last))[:, None]
        out[idx] = out[last] + t * (out[nxt] - out[last])
    gone = missing & ~fill
    out[gone] = np.nan
    return out, gone


def error_shape(prm):
    """(sigma in frames, amplitude) of the white noise whose Gaussian smoothing gives the ball's
    position error: RMS ball_error_m per axis and second difference with per-axis std
    ball_error_d2_m. White noise of amplitude a smoothed with sigma s has RMS a / sqrt(2 sqrt(pi) s)
    and second-difference std ~ a sqrt(3 / (8 sqrt(pi) s^5)), so s^2 = sqrt(3) / 2 * RMS / std."""
    sigma = math.sqrt(math.sqrt(3) / 2 * prm["ball_error_m"] / prm["ball_error_d2_m"])
    return sigma, prm["ball_error_m"] * math.sqrt(2 * math.sqrt(math.pi) * sigma)


def track_ball(det, missing, prm, z=None):
    """The model tracker on ball positions det [n, 2]: stretches without a ball shorter than
    fill_short_s filled in a straight line, everything smoothed with a centred Gaussian
    (ball_smooth_frames), plus a position error (error_shape, from unit white noise z [n, 2]) on the
    detected frames, straight through the filled ones. Returns (positions, still missing)."""
    path, gone = fill_straight(det, missing, prm["fill_short_s"])
    path = smooth_track(path, prm["ball_smooth_frames"])
    if z is not None and prm["ball_error_m"] > 0:
        sigma, amp = error_shape(prm)
        err, _ = fill_straight(smooth_track(z * amp, sigma), missing, prm["fill_short_s"])
        path = path + err
    return path, gone


def degrade_ball(clean, rng, prm, ctx=None):
    """ATD-like ball: long stretches without a ball where the real ATD has them (hidden_ball), the
    rest through the model tracker (track_ball): short losses filled in a straight line, a centred
    smoother and a slowly varying position error. No wrong detections: the real ATD has no jumps of
    metres. Stoppages come from the labels (that is what p_dead models), live stretches from random
    live frames; nothing else reads a label. If the source already lacks the ball (Metrica: during
    stoppages), those frames stay missing."""
    ball, graded = clean["ball"].astype(np.float64), clean["graded"]
    L, W = clean["pitch"]
    n = len(ball)
    ctx = ctx or ball_context(clean)
    already = float(np.mean(~ctx["present"][graded]))
    missing = hidden_ball(ctx, rng, prm, occlusions(clean, rng, prm["occlusion_share"], prm["fill_short_s"]))
    out, missing = track_ball(ball[:, :2], missing, prm, rng.normal(size=(n, 2)))
    missing |= ctx["present"] & ~graded & (rng.random(n) < 0.8)
    out[missing] = np.nan
    out[:, 0] = np.clip(out[:, 0], -0.05 * L, 1.05 * L)
    out[:, 1] = np.clip(out[:, 1], -0.05 * W, 1.05 * W)
    return out.astype(np.float32), already


def jittered(prm, rng):
    """A training seed's ball parameters: the calibrated ones, each scaled by a random factor
    (TRAIN_JITTER), and the uncalibrated ones drawn: the occlusion share from TRAIN_OCCLUSION and
    the size of the ball's position error from TRAIN_BALL_ERROR."""
    out = dict(prm)
    for key, (lo, hi) in TRAIN_JITTER.items():
        out[key] = float(prm[key] * rng.uniform(lo, hi))
    out["p_off"], out["p_dead"] = min(out["p_off"], 1.0), min(out["p_dead"], 1.0)
    out["occlusion_share"] = float(rng.uniform(*TRAIN_OCCLUSION))
    out["ball_error_m"] = float(rng.uniform(*TRAIN_BALL_ERROR))
    return out


BALL_LOSS = ("L1(shares by stretch length) + |missing - target| + |long per half - target| / target "
             "+ L1(shares of where the ball was last seen); the excess of SET PIECES preceded by a stretch "
             "over chance is reported, not fitted (~50 windows per match: too noisy)")


def ball_loss(pooled, target=REAL_BALL_GAPS):
    """Distance between the gap pattern of some matches and the real ATD's: L1 over the shares by
    stretch length, plus the error in the share without a ball, plus the relative error in long
    stretches per half, plus L1 over the shares of where the ball was last seen before a long
    stretch. The error in how much more often than chance a SET PIECES window is preceded by a long
    stretch ("set_pieces") is reported but left out: ~50 windows per match make it too noisy."""
    excess = lambda p, c: (p - c) / (1 - c)                                       # noqa: E731
    lengths = float(np.sum(np.abs(np.array(pooled["by_length"]) - target["by_length"])))
    long = abs(pooled["long_per_half"][0] - target["long_per_half"]) / target["long_per_half"]
    tags = abs(excess(pooled["preceded"], pooled["chance"]) -
               excess(target["set_pieces_preceded"], target["set_pieces_chance"]))
    where = float(np.sum(np.abs(np.array(pooled["where"]) / max(sum(pooled["where"]), 1) - target["where"])))
    miss = abs(pooled["missing"] - target["missing"])
    return {"lengths": round(lengths, 4), "missing_err": round(miss, 4), "long": round(long, 4),
            "set_pieces": round(tags, 4), "last_seen": round(where, 4),
            "loss": round(lengths + miss + long + where, 4)}


def _gap_scores(pooled):
    return {"missing": round(pooled["missing"], 4), "by_length": [round(x, 4) for x in pooled["by_length"]],
            "long_per_half": round(pooled["long_per_half"][0], 2), "preceded": round(pooled["preceded"], 4),
            "chance": round(pooled["chance"], 4),
            "where": [round(x / max(sum(pooled["where"]), 1), 4) for x in pooled["where"]]}


def calibrate_ball(train, draws=CALIBRATION_DRAWS):
    """p_off, p_dead, r_live and the live stretch lengths over BALL_GRID, scored with ball_loss on
    the IDSSE train matches (sorted by id), each degraded `draws` times and pooled: one draw per
    match leaves enough Monte Carlo noise to reorder settings within ~0.1 of loss. Each draw varies
    the parameters as a training seed does (jittered, occlusion share included), so that train
    reproduces the target on average. Common random numbers: every setting sees the same draws."""
    import itertools
    import ball_gap_diagnostic as bgd
    pre = []
    for k, c in enumerate(sorted(train, key=lambda c: c["match"])):
        ctx = ball_context(c)
        for r in range(draws):
            rng = np.random.default_rng([12345, k, r])
            windows = [(lo, hi) for lo, hi, _, _ in tag_instances(c, rng)]
            share = jittered(DEGRADE, np.random.default_rng([999, k, r]))["occlusion_share"]
            pre.append((ctx, occlusions(c, rng, share), windows, [(a, b) for _, a, b in c["halves"]],
                        c["ball"][:, :2], c["pitch"], [54321, k, r], [999, k, r]))
    grid = []
    for values in itertools.product(*BALL_GRID.values()):
        prm = dict(DEGRADE, **dict(zip(BALL_GRID, values)))
        results = []
        for ctx, occluded, windows, halves, ball, pitch, seed, jitter_seed in pre:
            missing = hidden_ball(ctx, np.random.default_rng(seed), jittered(prm, np.random.default_rng(jitter_seed)),
                                  occluded)
            missing &= ~short_stretches(missing, prm["fill_short_s"])
            results.append(bgd.measure(missing, halves, windows, ball, pitch, shift_draws=0))
        pooled = bgd.pool(results)
        grid.append(dict(zip(BALL_GRID, values), **ball_loss(pooled), **_gap_scores(pooled)))
    best = min(grid, key=lambda r: r["loss"])
    return {"target": REAL_BALL_GAPS, "loss": BALL_LOSS, "draws_per_match": draws,
            "grid": grid, "best": {k: best[k] for k in BALL_GRID}, "best_scores": best,
            "note": "Targets: tools/ball_gap_diagnostic.py on the challenge ATD -- aggregates without labels, "
                    "run by a person. Calibrated on the IDSSE train matches only."}


def calibrate_fill(refs, prm, draws=CALIBRATION_DRAWS):
    """fill_dead for the Metrica games over FILL_GRID: the share of the stoppages without a ball in
    the source that get an estimated position, scored with ball_loss on the val_metrica games (each
    degraded `draws` times), with the ball parameters calibrated on IDSSE (common random numbers)."""
    import ball_gap_diagnostic as bgd
    refs = sorted(refs, key=lambda c: c["match"])
    pre = []
    for k, c in enumerate(refs):
        gaps = source_gaps(c)
        for r in range(draws):
            rng = np.random.default_rng([12345, k, r])
            windows = [(lo, hi) for lo, hi, _, _ in tag_instances(c, rng)]
            pre.append((c, gaps, windows, [(a, b) for _, a, b in c["halves"]], [k, r]))
    grid = []
    for share in FILL_GRID:
        results = []
        for c, gaps, windows, halves, (k, r) in pre:
            filled = dict(c, ball=fill_source_gaps(c, np.random.default_rng([777, k, r]), share, gaps))
            occluded = occlusions(filled, np.random.default_rng([12345, k, r, 1]), prm["occlusion_share"])
            missing = hidden_ball(ball_context(filled), np.random.default_rng([54321, k, r]), prm, occluded)
            missing &= ~short_stretches(missing, prm["fill_short_s"])
            results.append(bgd.measure(missing, halves, windows, filled["ball"][:, :2], c["pitch"], shift_draws=0))
        pooled = bgd.pool(results)
        grid.append(dict(fill_dead=share, **ball_loss(pooled), **_gap_scores(pooled)))
    best = min(grid, key=lambda r: r["loss"])
    return {"loss": BALL_LOSS, "draws_per_match": draws, "grid": grid, "best": {"fill_dead": best["fill_dead"]},
            "best_scores": best, "source_gaps": [len(source_gaps(c)) for c in refs], "games": [c["match"] for c in refs],
            "note": "Calibrated on the val_metrica games only, the other ball parameters fixed at the IDSSE values."}


TRACK_LOSS = "sum over the fitted percentiles of |log(simulated / real)|"


def _d2_scores(hist, target, shown, fit):
    """The percentiles `shown` of a second-difference histogram, and the loss over those in `fit`."""
    import ball_gap_diagnostic as bgd
    vals = {"p%g" % q: round(bgd.d2_percentile(hist, q), 5) for q in shown}
    loss = sum(abs(math.log(max(vals["p%g" % q], 1e-6) / target["p%g" % q])) for q in fit)
    return vals, round(loss, 4)


def calibrate_tracking(train, prm, draws=CALIBRATION_DRAWS):
    """The model tracker, over TRACK_GRID_*, scored on the percentiles of the second difference
    (rounded to millimetres, as stored) on the IDSSE train matches (sorted by id) against the real
    ATD's (TRACK_LOSS). Ball: the smoother width and the second difference of the position error,
    on the frames left visible by the calibrated gaps, each match degraded `draws` times as a
    training seed (jittered: occlusion share and size of the error included); the best setting is
    also scored with the middle values (val/test). Players: the smoother width and the detection
    noise on every player's clean track; the slots' losses and swaps come later and the build
    reports the final percentiles. Common random numbers."""
    import itertools
    import ball_gap_diagnostic as bgd
    pre = []
    trains = sorted(train, key=lambda c: c["match"])
    for k, c in enumerate(trains):
        ctx = ball_context(c)
        for r in range(draws):
            p = jittered(prm, np.random.default_rng([999, k, r]))
            rng = np.random.default_rng([4242, k, r])
            missing = hidden_ball(ctx, rng, p, occlusions(c, rng, p["occlusion_share"], p["fill_short_s"]))
            pre.append((c["ball"][:, :2].astype(np.float64), missing, rng.normal(size=(c["n"], 2)),
                        [(a, b) for _, a, b in c["halves"]], p["ball_error_m"]))
    out = {"loss": TRACK_LOSS, "draws_per_match": draws,
           "note": "Targets: tools/ball_gap_diagnostic.py measure 8 on the challenge ATD -- aggregates without "
                   "labels, run by a person. Calibrated on the IDSSE train matches only."}
    def ball_hist(sigma, d2, middle=False):
        hist = 0
        for ball, missing, z, halves, error in pre:
            p = dict(prm, ball_smooth_frames=sigma, ball_error_d2_m=d2, ball_error_m=prm["ball_error_m"] if middle else error)
            xy, _ = track_ball(ball, missing, p, z)
            hist = hist + bgd.smoothness(np.round(xy * 1000) / 1000, halves)["d2_hist"]
        return hist

    grid = []
    for sigma, d2 in itertools.product(*TRACK_GRID_BALL.values()):
        vals, loss = _d2_scores(ball_hist(sigma, d2), REAL_BALL_GAPS["ball_d2"], bgd.BALL_D2_PERCENTILES, BALL_D2_FIT)
        grid.append(dict(zip(TRACK_GRID_BALL, (sigma, d2)), loss=loss, **vals))
    best = min(grid, key=lambda r: r["loss"])
    vals, loss = _d2_scores(ball_hist(best["ball_smooth_frames"], best["ball_error_d2_m"], middle=True),
                            REAL_BALL_GAPS["ball_d2"], bgd.BALL_D2_PERCENTILES, BALL_D2_FIT)
    out["ball"] = {"grid": grid, "best": {k: best[k] for k in TRACK_GRID_BALL}, "best_scores": best,
                   "middle_scores": dict(vals, loss=loss, ball_error_m=prm["ball_error_m"]),
                   "train_ball_error": list(TRAIN_BALL_ERROR), "target": REAL_BALL_GAPS["ball_d2"],
                   "fit": list(BALL_D2_FIT)}
    grid = []
    for sigma, noise in itertools.product(*TRACK_GRID_PLAYER.values()):
        hist = 0
        for k, c in enumerate(trains):
            raw = c["players"].astype(np.float64)
            z = np.random.default_rng([4343, k]).normal(size=raw.shape)
            xy = np.round(smooth_track(raw + noise * z, sigma) * 1000) / 1000
            hist = hist + bgd.smoothness(xy[:, 0], [(a, b) for _, a, b in c["halves"]], xy)["player_d2_hist"]
        vals, loss = _d2_scores(hist, REAL_BALL_GAPS["player_d2"], bgd.PLAYER_D2_PERCENTILES, PLAYER_D2_FIT)
        grid.append(dict(zip(TRACK_GRID_PLAYER, (sigma, noise)), loss=loss, **vals))
    best = min(grid, key=lambda r: r["loss"])
    out["players"] = {"grid": grid, "best": {k: best[k] for k in TRACK_GRID_PLAYER}, "best_scores": best,
                      "target": REAL_BALL_GAPS["player_d2"], "fit": list(PLAYER_D2_FIT)}
    return out


def degrade_players(clean, rng, prm):
    """Clean players -> 46 unreliable slots: lost and re-found in a new slot, swaps and ghosts, the
    positions through the model tracker (white detection noise, then a centred Gaussian smoother)."""
    raw = clean["players"].astype(np.float64)
    pos = smooth_track(raw + rng.normal(0, 1, raw.shape) * prm["player_detection_noise_m"], prm["player_smooth_frames"])
    team = clean["player_team"].astype(int)
    n, P = pos.shape[:2]
    out = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    slot_team = np.repeat([0, 1], SLOTS_PER_TEAM).astype(np.int8)
    free = [list(range(SLOTS_PER_TEAM)), list(range(SLOTS_PER_TEAM, 2 * SLOTS_PER_TEAM))]
    slot_of = np.full(P, -1, dtype=int)
    gap_left = np.zeros(P, dtype=int)
    ghosts = []
    same = team[:, None] == team[None, :]
    upper = np.triu(np.ones((P, P), dtype=bool), 1)
    gap_p = 1.0 / (prm["player_gap_s"] * FPS)

    def take(t):
        pool = free[t]
        return pool.pop(int(rng.integers(len(pool)))) if pool else -1

    def release(slot):
        if slot >= 0:
            free[slot_team[slot]].append(slot)

    for f in np.flatnonzero(np.any(~np.isnan(pos[:, :, 0]), axis=1)):
        p = pos[f].astype(np.float64)
        here = ~np.isnan(p[:, 0])
        for i in np.flatnonzero(~here & (slot_of >= 0)):
            release(slot_of[i])
            slot_of[i] = -1
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        crowd = np.nansum(d < 1.8, axis=1) - 1
        lose = (slot_of >= 0) & (rng.random(P) < prm["player_loss"] * (1 + 1.5 * np.minimum(crowd, 1)))
        for i in np.flatnonzero(lose):
            release(slot_of[i])
            slot_of[i], gap_left[i] = -1, 1 + rng.geometric(gap_p)
        waiting = (slot_of < 0) & here
        gap_left[waiting] -= 1
        for i in np.flatnonzero(waiting & (gap_left <= 0)):
            slot_of[i] = take(team[i])
        for i in np.flatnonzero((slot_of >= 0) & (rng.random(P) < prm["fragment"])):
            new = take(team[i])
            if new >= 0:
                release(slot_of[i])
                slot_of[i] = new
        tracked = (slot_of[:, None] >= 0) & (slot_of[None, :] >= 0) & upper
        with np.errstate(invalid="ignore"):
            for i, j in np.argwhere(tracked & same & (d < 1.2)):
                if rng.random() < prm["swap_same_team"]:
                    slot_of[i], slot_of[j] = slot_of[j], slot_of[i]
            for i, j in np.argwhere(tracked & ~same & (d < 1.0)):
                if rng.random() < prm["swap_cross_team"]:
                    slot_of[i], slot_of[j] = slot_of[j], slot_of[i]
        for i in np.flatnonzero((slot_of >= 0) & (rng.random(P) < prm["duplicate"])):
            slot = take(team[i])
            if slot >= 0:
                offset = rotate(np.array([1.0, 0.0]), rng.uniform(0, 2 * math.pi)) * rng.uniform(0.8, 2.0)
                ghosts.append([i, slot, offset, int(rng.integers(10, 41))])
        idx = np.flatnonzero(slot_of >= 0)
        out[f, slot_of[idx]] = p[idx]
        alive = []
        for g in ghosts:
            if here[g[0]] and g[3] > 0:
                out[f, g[1]] = p[g[0]] + g[2]
                g[3] -= 1
                alive.append(g)
            else:
                release(g[1])
        ghosts = alive
    return out, slot_team


# ==================================================================================================
# Synthetic Smart Tagging
# ==================================================================================================

def restarts_of(clean):
    """[(frame, team 0/1, type)] -- every D -> H/A change, typed by the source's own event within 2 s
    when it has them (Metrica: clean["restart_kinds"]), otherwise by where the ball is."""
    L, W = clean["pitch"]
    labels, ball = clean["labels"], clean["ball"]
    kinds = clean.get("restart_kinds") or []
    kind_frames = np.array([f for f, _ in kinds], dtype=np.int64)
    out = []
    for half, a, b in clean["halves"]:
        seg = labels[a:b + 1]
        first_live = True
        if seg[0] in ("H", "A"):
            out.append((a, "HA".index(seg[0]), "Kick Off (Start)"))
            first_live = False
        change = np.flatnonzero((seg[1:] != seg[:-1]) & (seg[:-1] == "D") & np.isin(seg[1:], ("H", "A"))) + 1
        for c in change:
            dead_len = c - (np.flatnonzero(seg[:c] != "D")[-1] + 1 if np.any(seg[:c] != "D") else 0)
            f = a + int(c)
            seen = np.flatnonzero(~np.isnan(ball[f:f + FPS, 0]))         # first sighting within 1 s
            x, y = (ball[f + seen[0], 0], ball[f + seen[0], 1]) if len(seen) else (math.nan, math.nan)
            j = int(np.argmin(np.abs(kind_frames - f))) if len(kind_frames) else -1
            if j >= 0 and abs(int(kind_frames[j]) - f) <= 2 * FPS:
                kind = kinds[j][1]
                if kind == "Kick Off":
                    kind = "Kick Off (Start)" if first_live else "Kick Off (After Goal)"
            elif math.isnan(x):
                kind = "Free Kick"
            elif math.hypot(x - L / 2, y - W / 2) < 3.0:
                kind = "Kick Off (Start)" if first_live else "Kick Off (After Goal)"
            elif min(math.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)) < CORNER_RADIUS_M:
                kind = "Corner Kick"
            elif y < 2.0 or y > W - 2.0:
                kind = "Throw-in"
            elif (x < 7.0 or x > L - 7.0) and abs(y - W / 2) < 11.0:
                kind = "Goal Kick"
            else:
                kind = "Free Kick"
            first_live = False
            if dead_len >= FPS or kind.startswith("Kick Off"):
                out.append((f, "HA".index(seg[c]), kind))
    return out


def tag_offset(rng, dist=TAG_OFFSET):
    """Seconds from the start of a SET PIECES window to its restart, drawn from TAG_OFFSET."""
    if dist["kind"] == "uniform":
        return float(rng.uniform(dist["lo_s"], dist["hi_s"]))
    edges, weights = np.asarray(dist["edges_s"], dtype=float), np.asarray(dist["weights"], dtype=float)
    k = int(rng.choice(len(weights), p=weights / weights.sum()))
    return float(rng.uniform(edges[k], edges[k + 1]))


def describe_offset(dist):
    """TAG_OFFSET in words, for the README."""
    if dist["kind"] == "uniform":
        text = "uniforme entre %g y %g s" % (dist["lo_s"], dist["hi_s"])
    else:
        w = np.asarray(dist["weights"], dtype=float)
        text = "histograma por tramos de %s s con pesos %s" % (
            ", ".join("%g–%g" % (a, b) for a, b in zip(dist["edges_s"][:-1], dist["edges_s"][1:])),
            ", ".join("%.2f" % x for x in w / w.sum()))
    return text + (" (**provisional**)" if dist.get("provisional") else "")


def jittered_offset(dist, rng):
    """A training seed's TAG_OFFSET: the histogram weights drawn from a Dirichlet centred on the
    calibrated ones and worth TAG_OFFSET_N observations."""
    w = np.asarray(dist["weights"], dtype=float)
    return dict(dist, weights=[round(float(x), 4) for x in rng.dirichlet(TAG_OFFSET_N * w / w.sum())])


def tag_instances(clean, rng, offset=TAG_OFFSET):
    """[(lo, hi, code, labels)] SET PIECES windows around the restarts, as the real file has them:
    TAG_WINDOW_S long with the restart TAG_OFFSET after the start, kick-offs TAG_KICKOFF_WINDOW_S
    long with the restart TAG_KICKOFF_OFFSET after it."""
    names = (clean["home"], clean["away"])
    half_name = {1: "1st Half", 2: "2nd Half"}

    def half_of(f):
        return next((h for h, a, b in clean["halves"] if a <= f <= b), 1)

    width = int(round(TAG_WINDOW_S * FPS))
    instances = []
    for f, team, kind in restarts_of(clean):
        if rng.random() >= TAG_RECALL[kind]:
            continue
        if kind == "Throw-in":
            label = "N/A"                                   # every throw-in in the real file
        else:
            label = names[team if rng.random() >= TAG_WRONG_TEAM else 1 - team]
        kickoff = kind.startswith("Kick Off")
        lo = max(f - int(round(tag_offset(rng, TAG_KICKOFF_OFFSET if kickoff else offset) * FPS)), 0)
        hi = lo + (int(round(TAG_KICKOFF_WINDOW_S * FPS)) if kickoff else width)
        instances.append((lo, hi, "SET PIECES", {"Team": label, "Half": half_name[half_of(f)], "Type": kind}))
    live = np.flatnonzero(np.isin(clean["labels"], ("H", "A")))
    for f in rng.choice(live, size=int(TAG_FALSE * len(instances)), replace=False):
        lo = max(int(f) - int(round(tag_offset(rng, offset) * FPS)), 0)
        instances.append((lo, lo + width, "SET PIECES",
                          {"Team": names[int(rng.integers(2))], "Half": half_name[half_of(int(f))], "Type": "Free Kick"}))
    instances.sort(key=lambda i: i[0])
    return instances


def synth_tagging(clean, rng, offset=TAG_OFFSET):
    instances = tag_instances(clean, rng, offset)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "  <ALL_INSTANCES>"]
    for k, (lo, hi, code, labels) in enumerate(instances, 1):
        lines.append("    <instance><ID>%d</ID><start>%.2f</start><end>%.2f</end><code>%s</code>" % (k, lo / FPS, hi / FPS, code))
        for group, text in labels.items():
            lines.append("      <label><group>%s</group><text>%s</text></label>" % (group, text))
        lines.append("    </instance>")
    lines += ["  </ALL_INSTANCES>", "</file>", ""]
    return "\n".join(lines), len(instances)


# ==================================================================================================
# Writing and loading
# ==================================================================================================

def to_cm(a):
    out = np.round(np.nan_to_num(a, nan=MISSING_CM / 100.0) * 100.0)
    return np.clip(out, MISSING_CM, 32767).astype(np.int16)


def from_cm(a):
    out = a.astype(np.float32) / 100.0
    out[a == MISSING_CM] = np.nan
    return out


def to_mm(a):
    out = np.round(np.nan_to_num(np.asarray(a, dtype=np.float64), nan=MISSING_MM / 1000.0) * 1000.0)
    return np.clip(out, MISSING_MM, 2 ** 31 - 1).astype(np.int32)


def from_mm(a):
    out = a.astype(np.float64) / 1000.0
    out[a == MISSING_MM] = np.nan
    return out.astype(np.float32)


def _write_labels(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (f, v) for f, v in enumerate(labels.tolist()))


def write_match(clean, out_dir, seeds, prm, harmonization, jitter=False):
    folder = os.path.join(out_dir, "matches", clean["match"])
    os.makedirs(folder, exist_ok=True)
    _write_labels(os.path.join(folder, "labels.csv"), clean["labels"])
    _write_labels(os.path.join(folder, "labels_raw.csv"), clean["labels_raw"])
    np.savez_compressed(os.path.join(folder, "truth.npz"), players_cm=to_cm(clean["players"]),
                        player_team=clean["player_team"], ball_cm=to_cm(clean["ball"]),
                        ball_status=clean["status"], ball_possession=clean["possession"])
    gt = possession_stats(clean["labels"], clean["halves"])
    gt_raw = possession_stats(clean["labels_raw"], clean["halves"])
    meta = {"match": clean["match"], "title": clean["title"], "division": clean["division"],
            "home": clean["home"], "away": clean["away"], "source": clean["source"],
            "citation": CITATION[clean["source"]], "fps": FPS, "n_frames": clean["n"], "pitch_m": list(clean["pitch"]),
            "halves": [list(h) for h in clean["halves"]], "flipped_second_half": clean["flipped_second_half"],
            "labels": clean["label_rule"] + "; X outside the halves",
            "harmonization": harmonization, "ground_truth": gt, "ground_truth_raw": gt_raw, "seeds": list(seeds)}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    rows = []
    ctx = ball_context(clean)
    gaps = source_gaps(clean) if prm["fill_dead"] > 0 else []
    for seed in seeds:
        key = zlib.crc32(clean["match"].encode())               # v1 used the first 4 bytes: "J03W" for every IDSSE match
        rng = np.random.default_rng([seed, key])
        t0 = time.time()
        prm_s = jittered(prm, np.random.default_rng([seed, key, 7])) if jitter else prm
        filled, ctx_s = clean, ctx
        if gaps:
            filled = dict(clean, ball=fill_source_gaps(clean, np.random.default_rng([seed, key, 11]), prm_s["fill_dead"], gaps))
            ctx_s = ball_context(filled)
        ball, _ = degrade_ball(filled, rng, prm_s, ctx_s)
        slots, slot_team = degrade_players(clean, rng, prm)
        offset = jittered_offset(TAG_OFFSET, np.random.default_rng([seed, key, 13])) if jitter else TAG_OFFSET
        tags, n_tags = synth_tagging(clean, rng, offset)
        sub = os.path.join(folder, "seed%d" % seed)
        os.makedirs(sub, exist_ok=True)
        np.savez_compressed(os.path.join(sub, "tracking.npz"), slots_mm=to_mm(slots), slot_team=slot_team,
                            ball_mm=to_mm(ball))
        with open(os.path.join(sub, "smart_tagging.xml"), "w", encoding="utf-8") as handle:
            handle.write(tags)
        g = clean["graded"]
        alive = g & np.isin(clean["labels"], ("H", "A"))
        dead = g & (clean["labels"] == "D")
        stats = {"ball_visible_graded": round(float(np.mean(~np.isnan(ball[g, 0]))), 4),
                 "ball_visible_live": round(float(np.mean(~np.isnan(ball[alive, 0]))), 4),
                 "ball_visible_dead": round(float(np.mean(~np.isnan(ball[dead, 0]))), 4),
                 "ball_missing_in_source_graded": round(float(np.mean(~ctx["present"][g])), 4),
                 "ball_filled_graded": round(float(np.mean((ctx_s["present"] & ~ctx["present"])[g])), 4),
                 "mean_tracked_objects_graded": round(float(np.mean(np.sum(~np.isnan(slots[g, :, 0]), axis=1))), 2),
                 "smart_tagging_instances": n_tags,
                 "targets_from_DATA_md": {"ball_visible_graded": 0.593, "mean_tracked_objects_graded": 20.5}}
        with open(os.path.join(sub, "degradation.json"), "w") as handle:
            json.dump({"seed": seed, "parameters": prm_s, "tag_recall": TAG_RECALL, "tag_wrong_team": TAG_WRONG_TEAM,
                       "tag_false": TAG_FALSE, "tag_window_s": TAG_WINDOW_S, "tag_offset": offset,
                       "tag_kickoff_window_s": TAG_KICKOFF_WINDOW_S, "tag_kickoff_offset": TAG_KICKOFF_OFFSET,
                       "stats": stats}, handle, indent=2)
        rows.append(dict(stats, seed=seed))
        print("    seed %d: ball visible %.1f%% (live %.1f%%, dead %.1f%%), %.1f objects, %d tags   (%.0f s)" % (
            seed, 100 * stats["ball_visible_graded"], 100 * stats["ball_visible_live"], 100 * stats["ball_visible_dead"],
            stats["mean_tracked_objects_graded"], n_tags, time.time() - t0))
    return meta, rows


def load_match(dataset_dir, match, seed=0, labels="harmonized"):
    """Everything a pipeline needs for one match, in the same units as the challenge loader:
    metres from the top-left corner, frames on the match's own timeline. labels: 'harmonized'
    (labels.csv, the default) or 'raw' (labels_raw.csv)."""
    folder = os.path.join(dataset_dir, "matches", match)
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    track = np.load(os.path.join(folder, "seed%d" % seed, "tracking.npz"))
    if "ball_mm" in track.files:                            # v2.1 on; v1 and v2 stored centimetres
        players, ball = from_mm(track["slots_mm"]), from_mm(track["ball_mm"])
    else:
        players, ball = from_cm(track["slots_cm"]), from_cm(track["ball_cm"])
    name = "labels.csv" if labels == "harmonized" else "labels_raw.csv"
    lab = np.loadtxt(os.path.join(folder, name), delimiter=",", skiprows=1, dtype=str)[:, 1]
    return {"meta": meta, "players": players, "slot_team": track["slot_team"],
            "ball": ball, "labels": lab,
            "smart_tagging": os.path.join(folder, "seed%d" % seed, "smart_tagging.xml"),
            "truth": os.path.join(folder, "truth.npz")}


# ==================================================================================================
# Verification: the official baselines on every match, scored with the official scorer
# ==================================================================================================

def nearest_player_codes(m):
    """The official baseline's raw signal on a loaded match: per frame, the team of the player nearest
    the ball (0 home, 1 away), -1 where the ball is missing. starter/baseline.smooth turns it into labels."""
    players, ball, team = m["players"], m["ball"], m["slot_team"]
    codes = np.full(len(ball), -1, dtype=np.int8)
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    for s in range(0, len(seen), 20000):
        idx = seen[s:s + 20000]
        d = np.linalg.norm(players[idx] - ball[idx, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        k = np.argmin(d, axis=1)
        ok = np.isfinite(d[np.arange(len(idx)), k])
        codes[idx[ok]] = team[k[ok]]
    return codes


def verify(out_dir, splits):
    import possession_metrics as pm
    from baseline import smooth

    windows, versions = (0, 25, 50, 125), ("harmonized", "raw")
    print("\nverify: nearest-player baselines, official scorer, per match (harmonised | raw labels)")
    rows = []
    for split, items in splits.items():
        for match, seed in items:
            m = load_match(out_dir, match, seed)
            raw = np.loadtxt(os.path.join(out_dir, "matches", match, "labels_raw.csv"), delimiter=",",
                             skiprows=1, dtype=str)[:, 1]
            codes = nearest_player_codes(m)
            gts = {"harmonized": pm.label_codes(m["labels"]), "raw": pm.label_codes(raw)}
            row = {"split": split, "match": match, "seed": seed, "source": m["meta"]["source"]}
            for w in windows:
                pred = pm.label_codes(smooth(codes, w))
                for v in versions:
                    row[v, w] = pm.evaluate(gts[v], pred)["spell_f1_mean"]
            rows.append(row)
            print("  %-12s %-8s s%d   %s" % (split, match, seed, "   ".join(
                "%s %.3f | %.3f" % ("smooth%d" % w if w else "raw", row["harmonized", w], row["raw", w]) for w in windows)))
    official = [50, 125, 25, 0]
    groups = {}
    for v in versions:
        groups[v] = {}
        for source in sorted({r["source"] for r in rows}):
            sel = [r for r in rows if r["source"] == source]
            mean = {w: float(np.mean([r[v, w] for r in sel])) for w in windows}
            order = sorted(mean, key=lambda w: -mean[w])
            groups[v][source] = {"mean": mean, "order": order, "same": order == official}
            print("  %-10s %-8s mean   raw %.3f   smooth25 %.3f   smooth50 %.3f   smooth125 %.3f   order: %s  (%s)" % (
                v, source, mean[0], mean[25], mean[50], mean[125],
                " > ".join("smooth%d" % w if w else "raw" for w in order),
                "same as the real match" if order == official else "real match: smooth50 > smooth125 > smooth25 > raw"))
    return groups


# ==================================================================================================
# Self-test source: the fake match
# ==================================================================================================

def clean_from_fake(truth_path):
    import games
    t = np.load(truth_path)
    L, W = games.GAMES[games.GAME]["pitch"]
    lab = t["label"]
    status = np.where(lab == 2, 0, np.where(lab >= 0, 1, -1)).astype(np.int8)
    possession = np.where(lab <= 1, lab, -1).astype(np.int8)
    ball = np.concatenate([t["ball"] * np.array([L, W]), np.where(t["ball_in_air"], 3.0, 0.1)[:, None]], axis=1)
    halves = [tuple(int(v) for v in h) for h in games.half_video_blocks(games.GAME)]
    clean = make_clean("FAKE0", "synthetic match (tools/make_fake_data.py)", "synthetic", "Home", "Away", (L, W),
                       halves, (t["players"] * np.array([L, W])).astype(np.float32), t["player_team"],
                       ball.astype(np.float32), status, possession)
    clean.update(source="fake", label_rule="the simulator's own labels")
    return clean


# ==================================================================================================
# README
# ==================================================================================================

README = """# Datasets v2.1 para More than a Hack 2027 (fase 1)

Generado por `tools/build_datasets.py` el {date}. Semillas de degradación: {seeds}.

## Qué cambia respecto a v2
v2 (`external/datasets_v2/`, construido con el commit 34dff27) y v1 (`external/datasets/`,
etiqueta git `v1-ball-degradation`) se quedan como estaban. `load_match()` no cambia (lee los tres);
las posiciones de `tracking.npz` pasan a **milímetros** (`ball_mm`, `slots_mm`, int32), porque un
tracker suavizado se mueve menos de un centímetro por frame.
- **Un tracker modelo** para el balón y los jugadores: las pérdidas cortas rellenas en línea recta
  y un suavizado gaussiano **centrado** (sin retraso: suponemos que el ATD es un producto offline),
  calibrado con los percentiles de la segunda diferencia de las posiciones medidos en el ATD del
  reto (la cola, los golpeos, fija la anchura). El balón lleva además un **error de posición** en
  los frames detectados (recto en los rellenos) que reproduce el cuerpo de esa distribución. Su
  tamaño **no se puede calibrar**: un error de 3 cm que cambia rápido y uno de 50 cm que cambia
  despacio dan los mismos percentiles, y la distancia al jugador más cercano tampoco lo separa
  (depende sobre todo del tracker: en Metrica el balón va pegado al pie). Así que cada semilla de
  train lo sortea entre {err_lo:g} y {err_hi:g} m por eje, y val y test usan el centro, {err:g} m.
  v2 añadía un ruido independiente por frame de 25 cm al balón y de 12 cm a los jugadores: una
  segunda diferencia ~60 y ~250 veces la real.
- **Sin detecciones falsas del balón**: el ATD real no tiene saltos de metros (p99,9 de la
  segunda diferencia: 0,1 m).
- **Oclusiones**: pérdidas breves (< {fill_short:g} s) que siempre se rellenan en línea recta. Su
  proporción no se puede medir en el real, así que **no se calibra**: cada semilla de train la
  sortea entre {occ_lo:g} y {occ_hi:g} de los frames evaluados (val y test: {occ:g}).
- **Tramos sin balón con el balón en juego** de {live_min:g} s o más (log-normal por encima). La
  calibración incluye la variación por semilla de train, para que la media de train reproduzca la
  del real (41 % de frames sin balón), y suma el error en el % sin balón a la pérdida.
- **Ventanas SET PIECES por tipo**: el tipo de cada reanudación sale de los eventos de Metrica; en
  IDSSE, de la posición del balón (córner a < {corner:g} m de un banderín). Faltas: ~1 de cada 7
  (como en el archivo real). Ver la tabla por tipo más abajo (en v2, la mitad de los córners y el
  doble de faltas).
- **Desplazamiento de la reanudación**: cada semilla de train sortea sus pesos de una Dirichlet
  centrada en `TAG_OFFSET` con {offset_n} observaciones.

## Lo que ya traía v2 (respecto a v1)
- **Ventanas SET PIECES de {tag_window:g} s** (los saques de centro, {kickoff_window:g} s), como en
  el archivo real (en v1, ~5 s todas). La reanudación cae a una distancia del inicio de la ventana
  sorteada de `TAG_OFFSET`: {tag_offset}. Sale del histograma del diagnóstico en el archivo real
  (fin de los tramos sin balón ≥ 10 s menos inicio de la ventana, menos lo esperado con las ventanas
  desplazadas al azar): todo el exceso cae entre −2,5 y 12,5 s, con el máximo en 5–7,5 s; en tramos
  de 5 s porque son unos 24 casos. El balón reaparece en la reanudación, así que el fin del tramo
  la marca con ~1 s de error.
- **Balón fuera del campo**: se esconde con probabilidad `p_off` (calibrada) en vez de siempre.
- **La calibración del balón** añade el reparto de dónde se vio el balón por última vez antes de
  un tramo sin balón de ≥ 10 s (fuera del campo, a < 2 m de la banda, a < 2 m de la línea de
  fondo, dentro), y las ventanas de 35 s cambian la tasa «por azar» de los SET PIECES precedidos.
- **Metrica**: una parte de las paradas en las que su tracking corregido no tiene balón recibe una
  posición estimada (la última vista, llevada en línea recta hasta el punto de la reanudación).
  Esa parte, `fill_dead`, se calibra con los mismos agregados sobre val_metrica.

Toda la calibración usa solo agregados sin etiquetas del ATD del reto (`REAL_BALL_GAPS`, medidos
con `tools/ball_gap_diagnostic.py` por una persona): nunca su posesión ni sus etiquetas.

## Fuentes, licencias y cita
- **IDSSE** (entrenamiento, validación y test): {cite_idsse}
  Datos de posición de la DFL a 25 fps con `BallStatus` y `BallPossession` en cada frame.
  Uso confirmado por los organizadores en el issue #16 del repo oficial, con tres condiciones:
  citarlo (aquí, en el README del repo y en el documento técnico); elegir todos los
  hiperparámetros y umbrales en estos partidos u otros, nunca en el del reto; y usar los 5 minutos
  públicos solo para comprobar el pipeline.
- **Metrica Sports sample data** (validación y test «de criterio»): {cite_metrica}
  Tracking corregido de Metrica y sus eventos etiquetados a mano. Su repositorio pide reconocer la
  fuente. **No trae etiquetas de posesión**: las derivamos de los eventos (`labels_from_events`),
  siguiendo la definición de posesión del PDF de Metrica, que coincide con la de CHALLENGE.md
  («la posesión no se pierde cuando la interferencia del rival no tiene un impacto significativo»).
  Son lo más parecido al criterio del reto que tenemos, pero son una aproximación: el reto deriva
  su verdad del tracking corregido, no de eventos.

## Particiones (por partido, nunca por frame)
{split_table}

- **train**: para entrenar.
- **val**: para elegir señales e hiperparámetros con datos abundantes.
- **val_metrica**: la validación que manda cuando el criterio de posesión importa (decodificador,
  duraciones). Si una idea gana en val pero pierde aquí, desconfiad de ella.
- **test** y **test_metrica**: no se miran hasta tener un candidato final.
- **comprobación** (no está aquí): los 5 minutos públicos del reto, en local, solo para ver que el
  pipeline funciona.

## Armonización de etiquetas
{harmonization_text}

## Qué hay en cada partido
```
matches/<PARTIDO>/meta.json              fuente, partes, regla de etiquetado, estadísticas
matches/<PARTIDO>/labels.csv             frame,label (H/A/D/X) armonizadas: las que se usan por defecto
matches/<PARTIDO>/labels_raw.csv         las etiquetas originales, para comprobar que una conclusión se sostiene
matches/<PARTIDO>/truth.npz              tracking limpio (para experimentos de oráculo)
matches/<PARTIDO>/seed<k>/tracking.npz   tracking degradado tipo ATD: 46 slots + balón
matches/<PARTIDO>/seed<k>/smart_tagging.xml   Smart Tagging sintético (formato SportsCode)
matches/<PARTIDO>/seed<k>/degradation.json    parámetros y estadísticas de la degradación
```
Posiciones en milímetros (int32 en `tracking.npz`, -2147483648 = sin dato; centímetros int16 en
`truth.npz`, -32768 = sin dato) desde la esquina superior izquierda (x a lo largo del campo, y
hacia abajo): las coordenadas 0–1 del ATD por el tamaño del campo. Cárgalo con
`load_match()` de `tools/build_datasets.py` (`labels="raw"` para las originales).

Línea de tiempo de cada partido: {pad_before} s sin juego, la 1ª parte, {halftime} s de descanso,
la 2ª parte y {pad_after} s sin juego. Fuera de las partes la verdad es X.

## En qué se parece y en qué no al partido del reto
Parecido, por construcción: 25 fps; 46 slots anónimos; jugadores que se pierden y vuelven en otro
slot, intercambios y duplicados (~20,5 objetos de media); la suavidad del tracking (percentiles de
la segunda diferencia del balón y de los jugadores); cambio de lado en el descanso;
Smart Tagging solo con SET PIECES, ventanas de {tag_window:g} s ({kickoff_window:g} s los saques de centro), con la
proporción por tipo del archivo real y `Team = N/A` en los saques de banda.

**El balón** sigue el patrón medido en el ATD del reto con `tools/ball_gap_diagnostic.py`
(agregados sin etiquetas, ejecutado por una persona: es el único uso del partido del reto en estos
datasets, y solo de la forma de sus huecos, nunca de la posesión). Allí no hay ningún tramo sin
balón de menos de 2 s y casi todo el balón ausente está en tramos de más de 10 s. Así que:
- las pérdidas cortas (< {fill_short:g} s) se **rellenan en línea recta** entre la última y la
  siguiente posición vista, y después el tracker modelo lo suaviza todo;
- el balón se esconde **fuera del campo** con probabilidad p_off, en las **paradas** con
  probabilidad p_dead y en **tramos con el balón en juego** de {live_min:g} s o más (r_live por
  minuto vivo, log-normal por encima);
- p_off, p_dead, r_live y las duraciones se calibran con esos agregados sobre los partidos de
  **train** de IDSSE; en train, cada semilla toma valores algo distintos alrededor de los
  calibrados;
- en **Metrica**, una parte fill_dead de las paradas sin balón en su tracking recibe una posición
  estimada (la última vista, llevada hasta la reanudación), calibrada sobre **val_metrica** con los
  demás parámetros fijos. `ball_degradation.json` guarda las dos rejillas.
{ball_text}

Distinto, y conviene recordarlo:
- En Metrica, las paradas que no se rellenan siguen sin balón, y la posición estimada es una
  línea recta: el balón real se queda quieto y luego lo llevan.
- Dónde cae la reanudación dentro de la ventana sale de unos 24 casos del archivo real, sin
  distinguir tipos de reanudación (por eso varía entre semillas de train); el resto de códigos del Smart Tagging real (fases, con la
  etiqueta Team del equipo con el balón) no se genera.
- La degradación es un modelo nuestro, no el tracker de Metrica.
- El suavizado es gaussiano y centrado: una suposición. El tamaño del error de posición del balón
  no lo sabemos (por eso varía entre semillas de train), y el de los jugadores lo suponemos nulo
  más allá del suavizado.
- La proporción de oclusiones (pérdidas breves rellenas) no se puede medir en el real: varía entre
  semillas sin calibrar.
- No hay fases tácticas en el Smart Tagging sintético, ni vídeo, ni homografía.
- Las etiquetas de Metrica salen de eventos: hay que revisar unos minutos a ojo antes de fiarse.

## Ventanas SET PIECES por tipo, cada 90 minutos evaluados
{tag_table}

## Estadísticas (etiquetas armonizadas)
{stats_table}
{verify_text}
"""


def tag_types_table(out_dir, metas, splits):
    """SET PIECES windows per Type per 90 graded minutes, per source over every match and seed, next
    to the real file's (REAL_BALL_GAPS), as a Markdown table."""
    counts = {}
    for items in splits.values():
        for match, seed in items:
            meta = next(m for m in metas if m["match"] == match)
            src = counts.setdefault(meta["source"], {"minutes": 0.0})
            src["minutes"] += sum(b + 1 - a for _, a, b in meta["halves"]) / FPS / 60.0
            for inst in ET.parse(os.path.join(out_dir, "matches", match, "seed%d" % seed, "smart_tagging.xml")).iter("instance"):
                kind = next((lab.findtext("text") for lab in inst.findall("label") if lab.findtext("group") == "Type"), "?")
                src[kind] = src.get(kind, 0) + 1
    real, minutes = REAL_BALL_GAPS["set_pieces_by_type"], REAL_BALL_GAPS["graded_minutes"]
    sources = sorted(counts)
    lines = ["| tipo | %s | ATD del reto |" % " | ".join(sources), "|---|%s---|" % ("---|" * len(sources))]
    for kind in list(real) + ["total"]:
        cells = ["%.1f" % (90.0 * (sum(c.get(k, 0) for k in real) if kind == "total" else c.get(kind, 0)) / c["minutes"])
                 for c in (counts[x] for x in sources)]
        r = sum(real.values()) if kind == "total" else real[kind]
        lines.append("| %s | %s | %.1f (%d) |" % (kind, " | ".join(cells), 90.0 * r / minutes, r))
    return "\n".join(lines)


def write_readme(out_dir, metas, rows, splits, seeds_per_split, harmonization, groups):
    split_of = {m: s for s, items in splits.items() for m, _ in items}
    ball_text = ""
    path = os.path.join(out_dir, "ball_degradation.json")
    if os.path.isfile(path):
        with open(path) as handle:
            bcal = json.load(handle)
        t, b = bcal["target"], bcal["best_scores"]
        cells = lambda r: " | ".join(["%.1f%%" % (100 * x) for x in r["by_length"]] + [  # noqa: E731
            "%.1f%%" % (100 * r["missing"]), "%.1f" % r["long_per_half"], "%.0f%% / %.0f%%" % (
                100 * r["preceded"], 100 * r["chance"]), " / ".join("%.0f%%" % (100 * x) for x in r["where"])])
        fill = bcal.get("metrica_fill")
        ball_text = ("\nCalibración (pérdida = %s): p_off %g, p_dead %g, r_live %g por min vivo, mediana %g s por "
                     "encima del mínimo, sigma %g%s. La fila de IDSSE incluye la variación por semilla de train.\n\n"
                     "| | < 1 s | 1–2 s | 2–5 s | 5–10 s | 10–30 s | > 30 s | sin balón | tramos ≥ 10 s por parte | "
                     "SET PIECES precedidos / azar | visto por última vez: fuera / banda / fondo / dentro |\n"
                     "|---|---|---|---|---|---|---|---|---|---|---|\n"
                     "| ATD del reto (objetivo) | %s |\n| IDSSE train, calibrado | %s |\n" % (
                         bcal["loss"], *[bcal["best"][k] for k in ("p_off", "p_dead", "r_live", "live_median_s",
                                                                   "live_sigma")],
                         "; en Metrica, fill_dead %g" % fill["best"]["fill_dead"] if fill else "",
                         cells(dict(t, preceded=t["set_pieces_preceded"], chance=t["set_pieces_chance"])), cells(b)))
        if fill:
            ball_text += "| val_metrica, sin rellenar | %s |\n| val_metrica, fill_dead %g | %s |\n" % (
                cells(fill["grid"][0]), fill["best"]["fill_dead"], cells(fill["best_scores"]))
        track = bcal.get("tracking")
        if track:
            ball_text += ("\nTracker modelo (pérdida = %s, ajustada en p%s del balón y p%s de los jugadores): balón, "
                          "suavizado σ = %g frames y un error de posición cuya segunda diferencia tiene %g m de "
                          "desviación por eje, con un RMS por eje **no calibrado** (train: uniforme entre %g y %g m; "
                          "val y test: %g m); jugadores, σ = %g frames y ruido %g m. Segunda diferencia, en metros "
                          "(la fila «calibrado» agrega las semillas de train; la de val/test usa el error del centro):\n\n" % (
                              track["loss"], "/".join("%g" % q for q in track["ball"]["fit"]),
                              "/".join("%g" % q for q in track["players"]["fit"]),
                              track["ball"]["best"]["ball_smooth_frames"], track["ball"]["best"]["ball_error_d2_m"],
                              *track["ball"]["train_ball_error"], track["ball"]["middle_scores"]["ball_error_m"],
                              track["players"]["best"]["player_smooth_frames"],
                              track["players"]["best"]["player_detection_noise_m"]))
            for part, name in (("ball", "balón"), ("players", "jugadores, por tracklet (limpio, antes de los slots)")):
                keys = list(track[part]["target"])
                ball_text += "| %s | %s |\n|---|%s\n| ATD del reto | %s |\n| IDSSE train, calibrado | %s |\n" % (
                    name, " | ".join(keys), "---|" * len(keys),
                    " | ".join("%.4f" % track[part]["target"][k] for k in keys),
                    " | ".join("%.4f" % track[part]["best_scores"][k] for k in keys))
                if "middle_scores" in track[part]:
                    ball_text += "| IDSSE train, error de val/test | %s |\n" % " | ".join(
                        "%.4f" % track[part]["middle_scores"][k] for k in keys)
                ball_text += "\n"
    split_table = "| partición | partidos | semillas |\n|---|---|---|\n" + "\n".join(
        "| %s | %s | %s |" % (s, ", ".join(sorted({m for m, _ in items})), seeds_per_split.get(s, 1))
        for s, items in splits.items() if items)
    head = ("| partido | fuente | partición | min evaluados | posesiones por min vivo | duración media (mediana) | "
            "< 1 s | % D | balón visible (vivo / muerto) | objetos |")
    stats = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for meta in metas:
        gt, raw = meta["ground_truth"], meta["ground_truth_raw"]
        r = [x for x in rows if x["match"] == meta["match"]]
        stats.append("| %s | %s | %s | %.0f | %.2f (bruto %.2f) | %.1f s (%.1f) | %.1f%% | %.1f%% | %.1f%% (%.0f / %.0f) | %.1f |" % (
            meta["match"], meta["source"], split_of.get(meta["match"], "-"), gt["graded_minutes"],
            gt["possessions_per_alive_min"], raw["possessions_per_alive_min"], gt["mean_possession_s"],
            gt["median_possession_s"], 100 * gt["share_under_1s"], 100 * gt["share"]["D"],
            100 * np.mean([x["ball_visible_graded"] for x in r]), 100 * np.mean([x["ball_visible_live"] for x in r]),
            100 * np.mean([x["ball_visible_dead"] for x in r]), np.mean([x["mean_tracked_objects_graded"] for x in r])))
    stats.append("\nReferencia del reto (DATA.md, partido completo): balón visible 59,3% de los frames evaluados, "
                 "20,5 objetos. Los 5 minutos públicos (17 posesiones, 4,2 por minuto vivo) son el arranque del "
                 "partido y no representan el partido entero.")
    verify_text = ""
    if groups:
        lines = ["\n## Comprobación",
                 "Baselines oficiales (más cercano + voto por mayoría), scorer oficial, media por fuente, con las "
                 "dos versiones de etiquetas. En el partido real el orden es smooth50 > smooth125 > smooth25 > raw."]
        for version, by_source in groups.items():
            for source, g in by_source.items():
                m = g["mean"]
                lines.append("- **%s**, etiquetas %s: raw %.3f · smooth25 %.3f · smooth50 %.3f · smooth125 %.3f → %s" % (
                    source, "armonizadas" if version == "harmonized" else "originales", m[0], m[25], m[50], m[125],
                    "mismo orden que el partido real" if g["same"] else
                    "orden distinto (%s)" % " > ".join("smooth%d" % w if w else "raw" for w in g["order"])))
        verify_text = "\n".join(lines) + "\n"
    h = harmonization
    htext = ("Dos limpiezas fijas para todas las fuentes: una parada de menos de %g s no es una parada, y un cambio "
             "de menos de %g s justo después de una reanudación es un parpadeo de los datos.\n\n" % (MIN_DEAD_S, MIN_RESTART_SPELL_S))
    if h["control_m"] > 0:
        htext += ("Y una regla solo para IDSSE: un tramo del rival que vuelve al mismo equipo sin parada es un toque, no "
                  "un cambio de posesión, si ningún jugador rival tuvo el balón a ≤ **%g m** durante ≥ **%g s** según el "
                  "tracking limpio (CHALLENGE.md: «un toque sin control», «un desvío», «un duelo aéreo sin control "
                  "posterior»)." % (h["control_m"], h["control_s"]))
        if "choice" in h:
            htext += (" En la rejilla de abajo, los ajustes de la diagonal (1 m / 0,2 s, 1,5 m / 0,4 s, 2 m / 0,5–0,6 s) "
                      "son equivalentes: sus pérdidas se diferencian menos que los dos partidos de referencia entre sí. "
                      "Dentro de ella se elige 1,5 m / 0,4 s por interpretación física: 1,5 m es el alcance de un "
                      "jugador más el error del tracking en la posición del jugador y del balón, y 0,4 s es el tiempo "
                      "aproximado de controlar el balón y jugarlo.")
    else:
        htext += "Regla de control: no aplicada."
    cal = h.get("calibration")
    if cal:
        rows = ["| d, c | posesiones por min vivo | mediana | < 1 s | < 2 s | < 5 s | KS | pérdida | %s |" %
                " | ".join("pérdida %s" % n for n in cal["references"]),
                "|---|---|---|---|---|---|---|---|%s" % ("---|" * len(cal["references"]))]
        ref = cal["grid"][0]["all"]
        for r in [dict(cal["no_merge"], label="sin regla")] + cal["grid"]:
            a = r.get("all", r)
            tag = r.get("label") or "%g m, %g s" % (r["control_m"], r["control_s"])
            if [r.get("control_m"), r.get("control_s")] == cal["chosen"]:
                tag = "**%s (elegida)**" % tag
            rows.append("| %s | %.2f | %.1f s | %.1f%% | %.1f%% | %.1f%% | %.3f | %.3f | %s |" % (
                tag, a["rate"], a["median_s"], 100 * a["under_1s"], 100 * a["under_2s"], 100 * a["under_5s"], a["ks"],
                a["loss"], " | ".join("%.3f" % r[n] if n in r else "" for n in cal["references"])))
        cross = "\n".join("- Elegida con %s: %g m, %g s → pérdida %.3f en %s y %.3f en %s (los dos partidos se diferencian "
                          "entre sí en %.3f)." % (x["fit_on"], x["picks"][0], x["picks"][1], x["loss_on_fit"], x["fit_on"],
                                                 x["loss_on_check"], x["check_on"], x["game_to_game_loss"])
                          for x in cal["cross_validation"])
        htext += ("\n\nCalibración: train de IDSSE contra la distribución de duraciones de val_metrica (%.2f posesiones por "
                  "minuto vivo). Pérdida = %s.\n\n%s\n\n%s" % (ref["rate_reference"], cal["loss"], "\n".join(rows), cross))
    seeds = ", ".join("%s ×%d" % (k, v) for k, v in seeds_per_split.items())
    text = README.format(date=time.strftime("%Y-%m-%d"), seeds=seeds, cite_idsse=CITATION["idsse"],
                         cite_metrica=CITATION["metrica"], split_table=split_table, harmonization_text=htext,
                         pad_before=PAD_BEFORE_S, halftime=HALFTIME_S, pad_after=PAD_AFTER_S,
                         fill_short=DEGRADE["fill_short_s"], ball_text=ball_text, tag_window=TAG_WINDOW_S,
                         occ_lo=TRAIN_OCCLUSION[0], occ_hi=TRAIN_OCCLUSION[1], occ=DEGRADE["occlusion_share"],
                         live_min=DEGRADE["live_min_s"], corner=CORNER_RADIUS_M, offset_n=TAG_OFFSET_N,
                         err_lo=TRAIN_BALL_ERROR[0], err_hi=TRAIN_BALL_ERROR[1], err=DEGRADE["ball_error_m"],
                         tag_offset=describe_offset(TAG_OFFSET), kickoff_window=TAG_KICKOFF_WINDOW_S,
                         stats_table="\n".join(stats), verify_text=verify_text,
                         tag_table=tag_types_table(out_dir, metas, splits))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(text)


# ==================================================================================================
# Main
# ==================================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "external", "idsse_raw"),
                        help="IDSSE XML files, downloaded if missing (default: external/idsse_raw)")
    parser.add_argument("--metrica-dir", default=os.path.join(ROOT, "external", "metrica_raw"),
                        help="Metrica sample files, downloaded if missing (default: external/metrica_raw)")
    parser.add_argument("--out", default=None, help="where to write the datasets (default: external/datasets_v2_1)")
    parser.add_argument("--matches", nargs="*", default=None, help="subset of match ids (default: all ten)")
    parser.add_argument("--train-seeds", type=int, default=2, help="degradations per training match")
    parser.add_argument("--eval-seeds", type=int, default=1, help="degradations per val/test match")
    parser.add_argument("--control-m", type=float, default=CONTROL_M,
                        help="control rule: distance to the ball in metres (0 = rule off)")
    parser.add_argument("--control-s", type=float, default=CONTROL_S, help="control rule: seconds within that distance")
    parser.add_argument("--verify", action="store_true", help="score the official baselines on every match")
    parser.add_argument("--selftest", action="store_true", help="fake match as IDSSE + the Metrica games")
    args = parser.parse_args()

    prm = dict(DEGRADE)
    t0 = time.time()
    split_of = {m: s for s, ms in DEFAULT_SPLITS.items() for m in ms}
    cleans = {}
    if args.selftest:
        out_dir = os.path.abspath(args.out or os.path.join(ROOT, "fake_data", "selftest_datasets"))
        print("self-test -> %s\nFAKE0  synthetic match, standing in for an IDSSE training match" % out_dir)
        cleans["FAKE0"] = clean_from_fake(os.path.join(ROOT, "fake_data", "seed0", "truth.npz"))
        split_of["FAKE0"] = "train"
        wanted = ["FAKE0"] + list(METRICA)
    else:
        out_dir = os.path.abspath(args.out or os.path.join(ROOT, "external", "datasets_v2_1"))
        wanted = args.matches or (list(IDSSE) + list(METRICA))

    for match in wanted:
        if match in cleans:
            continue
        t1 = time.time()
        if match in IDSSE:
            print("%s  %s  (%s)" % (match, IDSSE[match][0], split_of[match]))
            clean = parse_dfl(*fetch_idsse(match, os.path.abspath(args.raw_dir)), match)
        elif match in METRICA:
            print("%s  Metrica sample game %d  (%s)" % (match, METRICA[match], split_of[match]))
            clean = parse_metrica(match, fetch_metrica(match, os.path.abspath(args.metrica_dir)))
        else:
            raise SystemExit("unknown match %s" % match)
        gt = possession_stats(clean["labels"], clean["halves"])
        print("    parsed in %.0f s: %.0f min, %d possessions (%.2f per live min, mean %.1f s), D %.1f%%" % (
            time.time() - t1, gt["graded_minutes"], gt["possessions"], gt["possessions_per_alive_min"],
            gt["mean_possession_s"], 100 * gt["share"]["D"]))
        cleans[match] = clean

    # Labels: fixed clean-ups everywhere, then the control rule for DFL-labelled matches.
    for c in cleans.values():
        c["labels"] = clean_artefacts(c["labels"], c["halves"])
    dfl = [m for m, c in cleans.items() if c["source"] in ("idsse", "fake")]
    rule = {"min_dead_s": MIN_DEAD_S, "min_restart_spell_s": MIN_RESTART_SPELL_S, "rule": "control",
            "control_m": args.control_m, "control_s": args.control_s, "applies_to": "idsse",
            "control_gap_frames": CONTROL_GAP, "control_margin_frames": CONTROL_MARGIN}
    harmonization = dict(rule)
    if (args.control_m, args.control_s) == (CONTROL_M, CONTROL_S):
        harmonization["choice"] = CONTROL_CHOICE
    train = [cleans[m] for m in dfl if split_of[m] == "train"]
    refs = {m: c for m, c in cleans.items() if split_of.get(m) == "val_metrica"}
    if train and refs and args.control_m > 0:
        t1 = time.time()
        cal = calibrate_control(train, refs, (args.control_m, args.control_s))
        harmonization["calibration"] = cal
        print("\ncontrol rule: IDSSE train vs %s, loss = %s   (%.0f s)" % ("+".join(cal["references"]), cal["loss"], time.time() - t1))
        print("    %-14s %6s %7s %6s %6s %6s %6s %6s   %s" % ("d, c", "/min", "median", "<1 s", "<2 s", "<5 s", "KS", "loss",
                                                         "   ".join("loss %s" % n for n in cal["references"])))
        ref = distribution_loss(list(refs.values()), list(refs.values()))
        print("    %-14s %6.2f %6.1fs %5.1f%% %5.1f%% %5.1f%%" % ("reference", ref["rate"], ref["median_s"], 100 * ref["under_1s"],
                                                              100 * ref["under_2s"], 100 * ref["under_5s"]))
        for r in [dict(cal["no_merge"], label="no merge")] + cal["grid"]:
            a = r["all"] if "all" in r else r
            label = r.get("label") or "%.2f m %.1f s" % (r["control_m"], r["control_s"])
            print("    %-14s %6.2f %6.1fs %5.1f%% %5.1f%% %5.1f%% %6.3f %6.3f   %s%s" % (
                label, a["rate"], a["median_s"], 100 * a["under_1s"], 100 * a["under_2s"], 100 * a["under_5s"], a["ks"],
                a["loss"], "   ".join("%9.3f" % r[n] if n in r else "         " for n in cal["references"]),
                "   <- chosen" if [r.get("control_m"), r.get("control_s")] == cal["chosen"] else
                "   <- best" if [r.get("control_m"), r.get("control_s")] == cal["best_on_all"] else ""))
        for x in cal["cross_validation"]:
            print("    fit on %s -> %.2f m, %.1f s: loss %.3f there, %.3f on %s (the two games differ by %.3f)" % (
                x["fit_on"], x["picks"][0], x["picks"][1], x["loss_on_fit"], x["loss_on_check"], x["check_on"],
                x["game_to_game_loss"]))
    elif dfl:
        print("\n(no val_metrica games or no IDSSE train games: control rule applied without calibration report)")
    if args.control_m > 0:
        for m in dfl:
            ctrl = team_control(*ball_distances(cleans[m]), args.control_m, args.control_s)
            cleans[m]["labels"] = merge_uncontrolled(cleans[m]["labels"], cleans[m]["halves"], ctrl)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "label_harmonization.json"), "w") as handle:
        json.dump(harmonization, handle, indent=2)

    # Ball degradation: calibrated against the real ATD's gap pattern on the DFL train matches, then
    # the share of Metrica's stoppages without a ball to fill, on the val_metrica games.
    ball_train = [cleans[m] for m in dfl if split_of[m] == "train"]
    prm_metrica = dict(prm)
    if ball_train:
        t1 = time.time()
        bcal = calibrate_ball(ball_train)
        prm.update(bcal["best"])
        prm_metrica.update(bcal["best"])
        b = bcal["best_scores"]
        print("\nball degradation: %d settings on %d train matches vs the real ATD's aggregates  (%.0f s)" % (
            len(bcal["grid"]), len(ball_train), time.time() - t1))
        print("    best: %s   loss %.3f (lengths %.3f, missing %.3f, long per half %.3f, set pieces %.3f, last seen %.3f)" % (
            ", ".join("%s %g" % (k, prm[k]) for k in BALL_GRID), b["loss"], b["lengths"], b["missing_err"], b["long"],
            b["set_pieces"], b["last_seen"]))
        edges = [k for k, v in bcal["best"].items() if v in (min(BALL_GRID[k]), max(BALL_GRID[k]))
                 and not (k.startswith("p_") and v in (0.0, 1.0))]
        if edges:
            print("    WARNING: at the edge of BALL_GRID: %s -- widen the grid" % ", ".join(edges))
        rows = [("real ATD (target)", dict(REAL_BALL_GAPS, preceded=REAL_BALL_GAPS["set_pieces_preceded"],
                                           chance=REAL_BALL_GAPS["set_pieces_chance"])), ("IDSSE train, best", b)]
        refs_ball = [cleans[m] for m in cleans if split_of.get(m) == "val_metrica"]
        if refs_ball:
            t1 = time.time()
            fcal = calibrate_fill(refs_ball, prm)
            prm_metrica.update(fcal["best"])
            bcal["metrica_fill"] = fcal
            f = fcal["best_scores"]
            print("metrica fill: share of %s stoppage gaps in %s filled, over %d values  (%.0f s)" % (
                "+".join(str(n) for n in fcal["source_gaps"]), "+".join(fcal["games"]), len(fcal["grid"]),
                time.time() - t1))
            print("    best: fill_dead %g   loss %.3f (lengths %.3f, long per half %.3f, set pieces %.3f, last seen %.3f)" % (
                f["fill_dead"], f["loss"], f["lengths"], f["long"], f["set_pieces"], f["last_seen"]))
            rows += [("val_metrica, no fill", fcal["grid"][0]), ("val_metrica, best fill", f)]
        print("    %-24s %s" % ("", " ".join("%6s" % x for x in ("<1 s", "1-2", "2-5", "5-10", "10-30", ">30", "miss",
                                                              "long/h", "prec", "chance", "off", "touch", "goal", "in"))))
        for label, r in rows:
            print("    %-24s %s" % (label, " ".join(["%5.1f%%" % (100 * x) for x in r["by_length"]] +
                                                   ["%5.1f%%" % (100 * r["missing"]), "%6.1f" % r["long_per_half"],
                                                    "%5.1f%%" % (100 * r["preceded"]), "%5.1f%%" % (100 * r["chance"])] +
                                                   ["%5.1f%%" % (100 * x) for x in r["where"]])))
        t1 = time.time()
        tcal = calibrate_tracking(ball_train, prm)
        for part in ("ball", "players"):
            prm.update(tcal[part]["best"])
            prm_metrica.update(tcal[part]["best"])
        bcal["tracking"] = tcal
        print("model tracker: second difference (m) vs the real ATD, %d + %d settings  (%.0f s)" % (
            len(tcal["ball"]["grid"]), len(tcal["players"]["grid"]), time.time() - t1))
        for part, grid in (("ball", TRACK_GRID_BALL), ("players", TRACK_GRID_PLAYER)):
            b, target = tcal[part]["best_scores"], tcal[part]["target"]
            print("    %-8s best: %s   loss %.3f" % (part, ", ".join("%s %g" % (k, b[k]) for k in grid), b["loss"]))
            print("    %-8s %s" % ("", "  ".join("%7s" % k for k in target)))
            print("    %-8s %s" % ("real", "  ".join("%7.4f" % target[k] for k in target)))
            print("    %-8s %s" % ("best", "  ".join("%7.4f" % b[k] for k in target)))
            if "middle_scores" in tcal[part]:
                m = tcal[part]["middle_scores"]
                print("    %-8s %s   (ball_error_m %g, val/test)" % ("middle", "  ".join("%7.4f" % m[k] for k in target),
                                                                     m["ball_error_m"]))
            edges = [k for k in grid if b[k] in (min(grid[k]), max(grid[k])) and not (k.endswith("noise_m") and b[k] == 0)]
            if edges:
                print("    WARNING: at the edge of the tracker grid: %s -- widen the grid" % ", ".join(edges))
        with open(os.path.join(out_dir, "ball_degradation.json"), "w") as handle:
            json.dump(dict(bcal, train_jitter=TRAIN_JITTER, train_occlusion=TRAIN_OCCLUSION, tag_window_s=TAG_WINDOW_S,
                           tag_offset=TAG_OFFSET, tag_offset_n=TAG_OFFSET_N, tag_recall=TAG_RECALL,
                           corner_radius_m=CORNER_RADIUS_M, tag_kickoff_window_s=TAG_KICKOFF_WINDOW_S,
                           tag_kickoff_offset=TAG_KICKOFF_OFFSET,
                           fixed={k: v for k, v in DEGRADE.items()
                                  if k in ("fill_short_s", "occlusion_share", "dead_start_s", "dead_end_s",
                                           "live_min_s", "ball_error_m")},
                           train_ball_error=TRAIN_BALL_ERROR), handle, indent=1)
    else:
        print("\n(no DFL train matches: ball degradation with the default parameters, not calibrated)")

    seeds_per_split = {"train": args.train_seeds, "val": args.eval_seeds, "test": args.eval_seeds,
                       "val_metrica": args.eval_seeds, "test_metrica": args.eval_seeds}
    splits = {s: [] for s in seeds_per_split}
    metas, all_rows = [], []
    print()
    for match, clean in cleans.items():
        split = split_of[match]
        seeds = list(range(seeds_per_split[split]))
        print("%s  writing (%s)" % (match, split))
        meta, rows = write_match(clean, out_dir, seeds, prm_metrica if clean["source"] == "metrica" else prm,
                                 rule if clean["source"] != "metrica" else
                                 dict(rule, rule=None, note="reference labels: control rule not applied"),
                                 jitter=split == "train")
        metas.append(meta)
        all_rows += [dict(r, match=match) for r in rows]
        splits[split] += [[match, s] for s in seeds]
    cleans.clear()

    with open(os.path.join(out_dir, "splits.json"), "w") as handle:
        json.dump({k: v for k, v in splits.items() if v}, handle, indent=2)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["match", "source", "seed", "possessions", "possessions_per_alive_min", "possessions_per_alive_min_raw",
                         "mean_possession_s", "share_under_1s", "dead_share", "ball_visible_graded", "ball_visible_live",
                         "ball_visible_dead", "mean_tracked_objects_graded", "smart_tagging_instances"])
        for r in all_rows:
            meta = next(m for m in metas if m["match"] == r["match"])
            gt = meta["ground_truth"]
            writer.writerow([r["match"], meta["source"], r["seed"], gt["possessions"], gt["possessions_per_alive_min"],
                             meta["ground_truth_raw"]["possessions_per_alive_min"], gt["mean_possession_s"],
                             gt["share_under_1s"], gt["share"]["D"], r["ball_visible_graded"], r["ball_visible_live"],
                             r["ball_visible_dead"], r["mean_tracked_objects_graded"], r["smart_tagging_instances"]])
    groups = verify(out_dir, {s: v for s, v in splits.items() if v}) if args.verify else None
    write_readme(out_dir, metas, all_rows, splits, seeds_per_split, harmonization, groups)
    print("\ndatasets in %s  (%.0f s)" % (out_dir, time.time() - t0))


if __name__ == "__main__":
    main()