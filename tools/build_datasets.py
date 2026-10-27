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
deflection", "an aerial duel won without subsequent control"). The two values come from a grid
scored against the duration distribution of the Metrica VAL games (label_harmonization.json keeps
the grid, the loss and a cross-check between the two games) -- other games only, never the
challenge match. The raw labels are kept beside the harmonised ones (labels_raw.csv), so every
conclusion can be checked on both.

Neither source is the challenge data: both are clean tracking with player identities. Every match
is DEGRADED to look like the challenge ATD (46 anonymous slots, ball missing on ~41% of graded
frames, identity switches, noise) and a Smart Tagging file is SYNTHESISED from its restarts. Note
that Metrica's corrected tracking already has no ball during stoppages (visible ~97% of live
frames, ~0% of dead ones), while IDSSE has the ball everywhere and loses it to occlusions here.
Which pattern the challenge ATD follows is an open question -- see the dataset README.

    python tools/build_datasets.py --verify          # download, build and check
    python tools/build_datasets.py --selftest        # fake match (as IDSSE) + Metrica, no figshare needed

Defaults: IDSSE raw XML in external/idsse_raw/, Metrica raw files in external/metrica_raw/, the
datasets in external/datasets/ (add `external/` to .gitignore). Public data, not the challenge's:
fine for Claude Code to read.

Output (see the README.md written into it):

    README.md  splits.json  summary.csv  label_harmonization.json
    matches/<MATCH>/meta.json            source, timeline, label rule, ground-truth statistics
    matches/<MATCH>/labels.csv           frame,label (H/A/D/X), harmonised -- the official scorer's format
    matches/<MATCH>/labels_raw.csv       the labels before harmonisation
    matches/<MATCH>/truth.npz            the clean tracking, for oracle experiments and debugging
    matches/<MATCH>/seed<k>/tracking.npz ATD-like degraded tracking: 46 slots + ball
    matches/<MATCH>/seed<k>/smart_tagging.xml   synthetic, SportsCode XML like the real file
    matches/<MATCH>/seed<k>/degradation.json    parameters and the resulting statistics

Positions are int16 centimetres from the top-left corner (x along the pitch, y down), -32768 =
missing: the challenge ATD's 0-1 coordinates times the pitch size. Load with load_match().
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

FPS = 25
SLOTS_PER_TEAM = 23                    # 46 slots, as in the challenge ATD
PAD_BEFORE_S, HALFTIME_S, PAD_AFTER_S = 20, 60, 10
MISSING_CM = -32768

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

# Degradation. Targets from DATA.md: ball visible on 59.3% of graded frames, 20.5 tracked objects.
DEGRADE = {
    "ball_missing": 0.41,      # share of graded frames without a ball, counting any already missing
    "ball_noise_m": 0.25,
    "ball_false_rate": 0.01,
    "air_height_m": 1.8,
    "player_noise_m": 0.12,
    "player_loss": 0.0025,
    "player_gap_s": 1.2,
    "swap_same_team": 0.02,
    "swap_cross_team": 0.004,
    "fragment": 0.0004,
    "duplicate": 0.0002,
}
# Smart Tagging recall per restart type, from the real file's counts (DATA.md) vs a typical match.
TAG_RECALL = {"Throw-in": 0.55, "Corner Kick": 0.9, "Goal Kick": 0.55, "Free Kick": 0.25,
              "Kick Off (Start)": 1.0, "Kick Off (After Goal)": 1.0}
TAG_WRONG_TEAM = 0.05
TAG_FALSE = 0.03

# Label harmonisation: fixed artefact clean-ups for every source, plus the control rule for IDSSE.
MIN_DEAD_S = 1.0                       # a real stoppage lasts seconds
MIN_RESTART_SPELL_S = 0.2              # a flicker right after a restart is an artefact
CONTROL_M, CONTROL_S = 1.5, 0.5        # team choice; the grid and cross-check are in label_harmonization.json
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
    clean.update(source="metrica", label_rule="derived from Metrica's human event data (labels_from_events)")
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


def degrade_ball(clean, rng, prm):
    """Hide the ball where an automatic tracker loses it -- crowds, air, speed, off the pitch --
    add noise and a few wrong detections. Geometry only, never the labels. If the source already
    lacks the ball on some frames (Metrica: during stoppages), only the remainder is hidden."""
    ball, players, graded = clean["ball"].astype(np.float64), clean["players"], clean["graded"]
    L, W = clean["pitch"]
    n = len(ball)
    present = ~np.isnan(ball[:, 0])
    crowd = np.zeros(n)
    for s in range(0, n, 20000):
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(players[s:s + 20000] - ball[s:s + 20000, None, :2], axis=2)
            crowd[s:s + 20000] = np.nansum(d < 4.0, axis=1)
    speed = np.zeros(n)
    speed[1:] = np.nan_to_num(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * FPS)
    air = np.nan_to_num(ball[:, 2]) > prm["air_height_m"]
    with np.errstate(invalid="ignore"):
        off = present & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))
    score = 0.45 * crowd + 1.4 * air + 0.04 * np.minimum(speed, 30) + 3.0 * off + smooth_noise(rng, n, 21)
    already = float(np.mean(~present[graded]))
    extra = max(0.0, prm["ball_missing"] - already) / max(1e-9, 1.0 - already)
    hide = np.zeros(n, dtype=bool)
    if extra > 0:
        hide = score > np.quantile(score[graded & present], 1.0 - extra)
    missing = ~present | hide
    missing |= present & ~graded & (rng.random(n) < 0.8)

    out = ball[:, :2] + rng.normal(0, prm["ball_noise_m"], (n, 2))
    visible = np.flatnonzero(graded & ~missing)
    for start in rng.choice(visible, size=int(prm["ball_false_rate"] * len(visible) / 12), replace=False):
        end = min(start + int(rng.integers(5, 26)), n)
        offset = rotate(np.array([1.0, 0.0]), rng.uniform(0, 2 * math.pi)) * rng.uniform(6, 30)
        out[start:end] = ball[start, :2] + offset + rng.normal(0, 0.2, (end - start, 2))
        missing[start:end] = np.isnan(out[start:end, 0])
    out[missing] = np.nan
    out[:, 0] = np.clip(out[:, 0], -0.05 * L, 1.05 * L)
    out[:, 1] = np.clip(out[:, 1], -0.05 * W, 1.05 * W)
    return out.astype(np.float32), already


def degrade_players(clean, rng, prm):
    """Clean players -> 46 unreliable slots: lost and re-found in a new slot, swaps, ghosts, noise."""
    pos, team = clean["players"], clean["player_team"].astype(int)
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
        out[f, slot_of[idx]] = p[idx] + rng.normal(0, prm["player_noise_m"], (len(idx), 2))
        alive = []
        for g in ghosts:
            if here[g[0]] and g[3] > 0:
                out[f, g[1]] = p[g[0]] + g[2] + rng.normal(0, 0.2, 2)
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
    """[(frame, team 0/1, type)] -- every D -> H/A change, typed by where the ball is."""
    L, W = clean["pitch"]
    labels, ball = clean["labels"], clean["ball"]
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
            if math.isnan(x):
                kind = "Free Kick"
            elif math.hypot(x - L / 2, y - W / 2) < 3.0:
                kind = "Kick Off (Start)" if first_live else "Kick Off (After Goal)"
            elif min(math.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)) < 1.5:   # corner arc is 1 m
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


def synth_tagging(clean, rng):
    names = (clean["home"], clean["away"])
    half_name = {1: "1st Half", 2: "2nd Half"}

    def half_of(f):
        return next((h for h, a, b in clean["halves"] if a <= f <= b), 1)

    instances = []
    for f, team, kind in restarts_of(clean):
        if rng.random() >= TAG_RECALL[kind]:
            continue
        if kind == "Throw-in":
            label = "N/A"                                   # every throw-in in the real file
        else:
            label = names[team if rng.random() >= TAG_WRONG_TEAM else 1 - team]
        lo, hi = f - int(rng.uniform(2, 3) * FPS), f + int(rng.uniform(2, 3) * FPS)
        instances.append((lo, hi, "SET PIECES", {"Team": label, "Half": half_name[half_of(f)], "Type": kind}))
    live = np.flatnonzero(np.isin(clean["labels"], ("H", "A")))
    for f in rng.choice(live, size=int(TAG_FALSE * len(instances)), replace=False):
        lo = int(f) - int(rng.uniform(2, 3) * FPS)
        instances.append((lo, lo + int(rng.uniform(4, 6) * FPS), "SET PIECES",
                          {"Team": names[int(rng.integers(2))], "Half": half_name[half_of(int(f))], "Type": "Free Kick"}))
    instances.sort(key=lambda i: i[0])
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


def _write_labels(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (f, v) for f, v in enumerate(labels.tolist()))


def write_match(clean, out_dir, seeds, prm, harmonization):
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
    for seed in seeds:
        rng = np.random.default_rng([seed, int.from_bytes(clean["match"].encode(), "little") % (2 ** 32)])
        t0 = time.time()
        ball, already = degrade_ball(clean, rng, prm)
        slots, slot_team = degrade_players(clean, rng, prm)
        tags, n_tags = synth_tagging(clean, rng)
        sub = os.path.join(folder, "seed%d" % seed)
        os.makedirs(sub, exist_ok=True)
        np.savez_compressed(os.path.join(sub, "tracking.npz"), slots_cm=to_cm(slots), slot_team=slot_team,
                            ball_cm=to_cm(ball))
        with open(os.path.join(sub, "smart_tagging.xml"), "w", encoding="utf-8") as handle:
            handle.write(tags)
        g = clean["graded"]
        alive = g & np.isin(clean["labels"], ("H", "A"))
        dead = g & (clean["labels"] == "D")
        stats = {"ball_visible_graded": round(float(np.mean(~np.isnan(ball[g, 0]))), 4),
                 "ball_visible_live": round(float(np.mean(~np.isnan(ball[alive, 0]))), 4),
                 "ball_visible_dead": round(float(np.mean(~np.isnan(ball[dead, 0]))), 4),
                 "ball_missing_in_source_graded": round(already, 4),
                 "mean_tracked_objects_graded": round(float(np.mean(np.sum(~np.isnan(slots[g, :, 0]), axis=1))), 2),
                 "smart_tagging_instances": n_tags,
                 "targets_from_DATA_md": {"ball_visible_graded": 0.593, "mean_tracked_objects_graded": 20.5}}
        with open(os.path.join(sub, "degradation.json"), "w") as handle:
            json.dump({"seed": seed, "parameters": prm, "tag_recall": TAG_RECALL, "tag_wrong_team": TAG_WRONG_TEAM,
                       "tag_false": TAG_FALSE, "stats": stats}, handle, indent=2)
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
    name = "labels.csv" if labels == "harmonized" else "labels_raw.csv"
    lab = np.loadtxt(os.path.join(folder, name), delimiter=",", skiprows=1, dtype=str)[:, 1]
    return {"meta": meta, "players": from_cm(track["slots_cm"]), "slot_team": track["slot_team"],
            "ball": from_cm(track["ball_cm"]), "labels": lab,
            "smart_tagging": os.path.join(folder, "seed%d" % seed, "smart_tagging.xml"),
            "truth": os.path.join(folder, "truth.npz")}


# ==================================================================================================
# Verification: the official baselines on every match, scored with the official scorer
# ==================================================================================================

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

README = """# Datasets para More than a Hack 2027 (fase 1)

Generado por `tools/build_datasets.py` el {date}. Semillas de degradación: {seeds}.

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
Posiciones en centímetros (int16) desde la esquina superior izquierda (x a lo largo del campo, y
hacia abajo), -32768 = sin dato: las coordenadas 0–1 del ATD por el tamaño del campo. Cárgalo con
`load_match()` de `tools/build_datasets.py` (`labels="raw"` para las originales).

Línea de tiempo de cada partido: {pad_before} s sin juego, la 1ª parte, {halftime} s de descanso,
la 2ª parte y {pad_after} s sin juego. Fuera de las partes la verdad es X.

## En qué se parece y en qué no al partido del reto
Parecido, por construcción: 25 fps; 46 slots anónimos; balón ausente en ~41% de los frames
evaluados; jugadores que se pierden y vuelven en otro slot, intercambios, duplicados y ruido
(~20,5 objetos de media); cambio de lado en el descanso; Smart Tagging solo con SET PIECES, con la
proporción por tipo del archivo real y `Team = N/A` en los saques de banda.

Distinto, y conviene recordarlo:
- **Dónde se pierde el balón.** En IDSSE el balón existe siempre y la degradación lo esconde por
  motivos físicos (gente cerca, aire, velocidad, fuera del campo), casi siempre con el balón en
  juego. En Metrica el tracking corregido ya no tiene balón durante las paradas, así que allí
  «no se ve el balón» casi equivale a «balón muerto». No sabemos cuál de los dos patrones sigue el
  ATD del reto: medidlo en el partido real sin etiquetas (duración de los tramos sin balón) antes
  de confiar en una señal basada en eso.
- La degradación es un modelo nuestro, no el tracker de Metrica.
- No hay fases tácticas en el Smart Tagging sintético, ni vídeo, ni homografía.
- Las etiquetas de Metrica salen de eventos: hay que revisar unos minutos a ojo antes de fiarse.

## Estadísticas (etiquetas armonizadas)
{stats_table}
{verify_text}
"""


def write_readme(out_dir, metas, rows, splits, seeds_per_split, harmonization, groups):
    split_of = {m: s for s, items in splits.items() for m, _ in items}
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
                  "posterior»). Los dos valores los fija el equipo; la calibración de abajo muestra cómo quedan frente "
                  "al resto de la rejilla." % (h["control_m"], h["control_s"]))
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
                         stats_table="\n".join(stats), verify_text=verify_text)
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
    parser.add_argument("--out", default=None, help="where to write the datasets (default: external/datasets)")
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
        out_dir = os.path.abspath(args.out or os.path.join(ROOT, "external", "datasets"))
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

    seeds_per_split = {"train": args.train_seeds, "val": args.eval_seeds, "test": args.eval_seeds,
                       "val_metrica": args.eval_seeds, "test_metrica": args.eval_seeds}
    splits = {s: [] for s in seeds_per_split}
    metas, all_rows = [], []
    print()
    for match, clean in cleans.items():
        split = split_of[match]
        seeds = list(range(seeds_per_split[split]))
        print("%s  writing (%s)" % (match, split))
        meta, rows = write_match(clean, out_dir, seeds, prm,
                                 rule if clean["source"] != "metrica" else
                                 dict(rule, rule=None, note="reference labels: control rule not applied"))
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