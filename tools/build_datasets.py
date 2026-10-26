#!/usr/bin/env python3
"""Build train / validation / test datasets in the shape of the challenge, from public matches.

Source: the IDSSE dataset (Bassek, Weber, Rein & Memmert, Scientific Data 2025), seven full matches
of the 2022/23 Bundesliga and 2. Bundesliga with DFL position data at 25 fps, licensed CC BY 4.0.
The organisers confirmed it may be used (metrica-sports/more-than-a-hack-2027, issue #16) on three
conditions: cite it, choose every hyperparameter and threshold on IDSSE or other games, and use the
5 public minutes of the provided game only to check the pipeline.

Why IDSSE: every ball frame carries BallStatus (alive/dead) and BallPossession (home/away), taken
from corrected tracking. That is how the challenge's own ground truth is defined (CHALLENGE.md:
"D where the corrected tracking marks the ball dead, otherwise the team the tracking names as
owning it"), so the labels mean the same thing.

What it is NOT: the challenge data. IDSSE is clean optical tracking with player identities and the
ball almost always visible; the challenge ATD comes from video, uncorrected, with 46 anonymous
slots and the ball missing on 41% of graded frames, and it comes with Smart Tagging. So every
match is DEGRADED to look like the ATD, and a Smart Tagging file is SYNTHESISED from its restarts,
with the defects the real file has. Both are assumptions -- the main risk of training on this data
-- and they are written down in each match's degradation.json and in the dataset README.

    python tools/build_datasets.py --verify          # download, build and check (~15-20 min, ~2.6 GB raw)
    python tools/build_datasets.py --selftest        # the same code on fake_data/seed0, no download

By default everything goes to external/ inside the repository (add `external/` to .gitignore):
the raw XML to external/idsse_raw/ and the datasets to external/datasets/. This is public data,
not the challenge's, so it is fine for Claude Code to read it.

The raw files are downloaded from figshare (the official host) or, failing that, from the Hugging
Face mirror pysport/idsse-data. Already-downloaded files are reused: put them in --raw-dir as
<MATCH>_meta.xml and <MATCH>_positions.xml.

Output (one folder; see the README.md written into it):

    README.md  splits.json  summary.csv
    matches/<MATCH>/meta.json            match facts, timeline, ground-truth statistics
    matches/<MATCH>/labels.csv           frame,label (H/A/D/X) -- the official scorer's format
    matches/<MATCH>/truth.npz            the clean tracking, for oracle experiments and debugging
    matches/<MATCH>/seed<k>/tracking.npz ATD-like degraded tracking: 46 slots + ball
    matches/<MATCH>/seed<k>/smart_tagging.xml   synthetic, SportsCode XML like the real file
    matches/<MATCH>/seed<k>/degradation.json    parameters and the resulting statistics

Positions are stored as int16 centimetres from the top-left corner (x along the pitch, y down),
-32768 meaning missing: exactly the challenge ATD's 0-1 coordinates times the pitch size.
Use load_match() below, or tools/mtah_data.py, rather than reading the arrays by hand.
"""

import argparse
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

# Split by MATCH, never by frame. Both top-flight games are split between train and val; the five
# Düsseldorf home games are spread over all three splits. Features are team-agnostic, so a club
# appearing in two splits is not a leak.
DEFAULT_SPLITS = {"train": ["J03WPY", "J03WOH", "J03WQQ", "J03WN1"],
                  "val": ["J03WOY", "J03WMX"],
                  "test": ["J03WR9"]}

CITATION = ("Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). An integrated dataset of "
            "spatiotemporal and event data in elite soccer. Scientific Data, 12. "
            "Licensed CC BY 4.0.")

# Degradation defaults. Targets come from DATA.md (ball visible on 59.3% of graded frames, 20.5
# tracked objects on average, 46 slots); the rates are the same as tools/make_fake_data.py, which
# lands on those targets.
DEGRADE = {
    "ball_missing": 0.41,      # share of graded frames without a ball (DATA.md)
    "ball_noise_m": 0.25,
    "ball_false_rate": 0.01,   # share of visible frames covered by a wrong ball detection
    "air_height_m": 1.8,       # a ball above this is "in the air": harder to track
    "player_noise_m": 0.12,
    "player_loss": 0.0025,     # per-frame chance of losing a player (x up to 2.5 in crowds)
    "player_gap_s": 1.2,
    "swap_same_team": 0.02,
    "swap_cross_team": 0.004,
    "fragment": 0.0004,
    "duplicate": 0.0002,
}
# Smart Tagging: share of restarts of each kind that get a SET PIECES tag. The real file has 26
# throw-ins, 12 corners, 10 goal kicks, 5 free kicks and 6 kick-offs (DATA.md); a typical match
# has roughly 45, 10, 18, 25 and 4-8, which is where these recalls come from.
TAG_RECALL = {"Throw-in": 0.55, "Corner Kick": 0.9, "Goal Kick": 0.55, "Free Kick": 0.25,
              "Kick Off (Start)": 1.0, "Kick Off (After Goal)": 1.0}
TAG_WRONG_TEAM = 0.05
TAG_FALSE = 0.03


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


# ==================================================================================================
# Parse DFL XML into a clean match
# ==================================================================================================

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
    ranges = {}
    for s in sections:
        ball_set = [v for k, v in sets.items() if k[0] == s and k[1] == "BALL"][0]
        ranges[s] = (int(ball_set["N"].min()), int(ball_set["N"].max()))
    offsets, pos = {}, PAD_BEFORE_S * FPS
    halves = []
    for i, s in enumerate(sections):
        first, last = ranges[s]
        offsets[s] = pos - first
        halves.append((i + 1, pos, pos + last - first))
        pos += last - first + 1 + (HALFTIME_S * FPS if i == 0 else 0)
    n = pos + PAD_AFTER_S * FPS

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
    return make_clean(match, title, division, home_name, away_name, (L, W), halves,
                      players, player_team, ball, status, possession)


def make_clean(match, title, division, home, away, pitch, halves, players, player_team, ball, status, possession):
    """Labels from BallStatus/BallPossession, and sides that swap at half-time like a real camera."""
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
            "possession": possession, "labels": labels, "graded": graded, "flipped_second_half": flipped}


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


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
    add noise and a few wrong detections. Driven by geometry only, never by the labels."""
    ball, players, graded = clean["ball"].astype(np.float64), clean["players"], clean["graded"]
    L, W = clean["pitch"]
    n = len(ball)
    present = ~np.isnan(ball[:, 0])
    crowd = np.zeros(n)
    for s in range(0, n, 20000):
        d = np.linalg.norm(players[s:s + 20000] - ball[s:s + 20000, None, :2], axis=2)
        crowd[s:s + 20000] = np.nansum(d < 4.0, axis=1)
    speed = np.zeros(n)
    speed[1:] = np.nan_to_num(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * FPS)
    air = np.nan_to_num(ball[:, 2]) > prm["air_height_m"]
    with np.errstate(invalid="ignore"):
        off = present & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))
    score = 0.45 * crowd + 1.4 * air + 0.04 * np.minimum(speed, 30) + 3.0 * off + smooth_noise(rng, n, 21)
    threshold = np.quantile(score[graded & present], 1.0 - prm["ball_missing"])
    missing = ~present | (score > threshold)
    missing |= present & ~graded & (rng.random(n) < 0.8)

    out = ball[:, :2] + rng.normal(0, prm["ball_noise_m"], (n, 2))
    visible = np.flatnonzero(graded & ~missing)
    for start in rng.choice(visible, size=int(prm["ball_false_rate"] * len(visible) / 12), replace=False):
        end = min(start + int(rng.integers(5, 26)), n)
        offset = rotate(np.array([1.0, 0.0]), rng.uniform(0, 2 * math.pi)) * rng.uniform(6, 30)
        out[start:end] = ball[start, :2] + offset + rng.normal(0, 0.2, (end - start, 2))
        missing[start:end] = False
    out[missing] = np.nan
    out[:, 0] = np.clip(out[:, 0], -0.05 * L, 1.05 * L)
    out[:, 1] = np.clip(out[:, 1], -0.05 * W, 1.05 * W)
    return out.astype(np.float32)


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
            x, y = ball[f, 0], ball[f, 1]
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


def possession_stats(labels, halves):
    lengths, per_half = [], []
    for _, a, b in halves:
        seg = labels[a:b + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], cut, [len(seg)]))
        runs = [(int(q - p), seg[p]) for p, q in zip(bounds[:-1], bounds[1:])]
        team_runs = [r for r, v in runs if v in ("H", "A")]
        per_half.append(len(team_runs))
        lengths += team_runs
    g = np.concatenate([labels[a:b + 1] for _, a, b in halves])
    minutes = len(g) / FPS / 60.0
    return {"possessions": int(sum(per_half)), "per_half": per_half,
            "mean_possession_s": round(float(np.mean(lengths)) / FPS, 2) if lengths else 0.0,
            "possessions_per_min": round(sum(per_half) / minutes, 2),
            "share": {k: round(float(np.mean(g == k)), 4) for k in ("H", "A", "D")},
            "graded_minutes": round(minutes, 1)}


def write_match(clean, out_dir, seeds, prm):
    folder = os.path.join(out_dir, "matches", clean["match"])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "labels.csv"), "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (f, v) for f, v in enumerate(clean["labels"].tolist()))
    np.savez_compressed(os.path.join(folder, "truth.npz"), players_cm=to_cm(clean["players"]),
                        player_team=clean["player_team"], ball_cm=to_cm(clean["ball"]),
                        ball_status=clean["status"], ball_possession=clean["possession"])
    gt = possession_stats(clean["labels"], clean["halves"])
    meta = {"match": clean["match"], "title": clean["title"], "division": clean["division"],
            "home": clean["home"], "away": clean["away"], "source": "IDSSE (DFL position data)",
            "citation": CITATION, "fps": FPS, "n_frames": clean["n"], "pitch_m": list(clean["pitch"]),
            "halves": [list(h) for h in clean["halves"]], "flipped_second_half": clean["flipped_second_half"],
            "labels": "D where BallStatus=0, else the BallPossession team; X outside the halves",
            "ground_truth": gt, "seeds": list(seeds)}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng([seed, int.from_bytes(clean["match"].encode(), "little") % (2 ** 32)])
        t0 = time.time()
        ball = degrade_ball(clean, rng, prm)
        slots, slot_team = degrade_players(clean, rng, prm)
        tags, n_tags = synth_tagging(clean, rng)
        sub = os.path.join(folder, "seed%d" % seed)
        os.makedirs(sub, exist_ok=True)
        np.savez_compressed(os.path.join(sub, "tracking.npz"), slots_cm=to_cm(slots), slot_team=slot_team,
                            ball_cm=to_cm(ball))
        with open(os.path.join(sub, "smart_tagging.xml"), "w", encoding="utf-8") as handle:
            handle.write(tags)
        g = clean["graded"]
        stats = {"ball_visible_graded": round(float(np.mean(~np.isnan(ball[g, 0]))), 4),
                 "mean_tracked_objects_graded": round(float(np.mean(np.sum(~np.isnan(slots[g, :, 0]), axis=1))), 2),
                 "smart_tagging_instances": n_tags,
                 "targets_from_DATA_md": {"ball_visible_graded": 0.593, "mean_tracked_objects_graded": 20.5}}
        with open(os.path.join(sub, "degradation.json"), "w") as handle:
            json.dump({"seed": seed, "parameters": prm, "tag_recall": TAG_RECALL, "tag_wrong_team": TAG_WRONG_TEAM,
                       "tag_false": TAG_FALSE, "stats": stats}, handle, indent=2)
        rows.append(dict(stats, seed=seed))
        print("    seed %d: ball visible %.1f%%, %.1f objects, %d tags   (%.0f s)" % (
            seed, 100 * stats["ball_visible_graded"], stats["mean_tracked_objects_graded"], n_tags, time.time() - t0))
    return meta, rows


def load_match(dataset_dir, match, seed=0):
    """Everything a pipeline needs for one match, in the same units as the challenge loader:
    metres from the top-left corner, frames on the match's own timeline."""
    folder = os.path.join(dataset_dir, "matches", match)
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    track = np.load(os.path.join(folder, "seed%d" % seed, "tracking.npz"))
    labels = np.loadtxt(os.path.join(folder, "labels.csv"), delimiter=",", skiprows=1, dtype=str)[:, 1]
    return {"meta": meta, "players": from_cm(track["slots_cm"]), "slot_team": track["slot_team"],
            "ball": from_cm(track["ball_cm"]), "labels": labels,
            "smart_tagging": os.path.join(folder, "seed%d" % seed, "smart_tagging.xml"),
            "truth": os.path.join(folder, "truth.npz")}


# ==================================================================================================
# Verification: the official baselines on every match, scored with the official scorer
# ==================================================================================================

def verify(out_dir, splits):
    import possession_metrics as pm
    from baseline import smooth

    print("\nverify: nearest-player baselines, official scorer, per match")
    table = []
    for split, items in splits.items():
        for match, seed in items:
            m = load_match(out_dir, match, seed)
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
            gt = pm.label_codes(m["labels"])
            row = {"split": split, "match": match, "seed": seed}
            for w in (0, 25, 50, 125):
                row["w%d" % w] = pm.evaluate(gt, pm.label_codes(smooth(codes, w)))["spell_f1_mean"]
            table.append(row)
            print("  %-5s %s s%d   raw %.3f   smooth25 %.3f   smooth50 %.3f   smooth125 %.3f" % (
                split, match, seed, row["w0"], row["w25"], row["w50"], row["w125"]))
    mean = {w: float(np.mean([r["w%d" % w] for r in table])) for w in (0, 25, 50, 125)}
    order = sorted(mean, key=lambda w: -mean[w])
    official = [50, 125, 25, 0]
    print("  mean        raw %.3f   smooth25 %.3f   smooth50 %.3f   smooth125 %.3f" % (mean[0], mean[25], mean[50], mean[125]))
    print("  ordering on these datasets: %s" % " > ".join("smooth%d" % w if w else "raw" for w in order))
    print("  ordering on the real match: smooth50 > smooth125 > smooth25 > raw  -> %s" % (
        "SAME: the datasets rank methods like the real match" if order == official else
        "DIFFERENT: treat conclusions drawn on these datasets with caution"))
    return table, mean, order == official


# ==================================================================================================
# Self-test without downloading: the fake match, pushed through the same code path
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
    return make_clean("FAKE0", "synthetic match (tools/make_fake_data.py)", "synthetic", "Home", "Away", (L, W),
                      halves, (t["players"] * np.array([L, W])).astype(np.float32), t["player_team"],
                      ball.astype(np.float32), status, possession)


# ==================================================================================================
# README
# ==================================================================================================

README = """# Datasets para More than a Hack 2027 (fase 1)

Generado por `tools/build_datasets.py` el {date}. Semillas de degradación: {seeds}.

## Fuente y licencia
Partidos del dataset **IDSSE**: {citation}
Datos de posición de la DFL a 25 fps, con `BallStatus` (balón vivo/muerto) y `BallPossession`
(equipo) en cada frame, derivados de tracking corregido. Es la misma definición que la verdad del
reto (CHALLENGE.md). Uso confirmado por los organizadores en el issue #16 del repo oficial, con
tres condiciones: citar el dataset (aquí, en el README del repo y en el documento técnico); elegir
todos los hiperparámetros y umbrales en estos partidos u otros, nunca en el del reto; y usar los
5 minutos públicos solo para comprobar el pipeline.

## Particiones (por partido, nunca por frame)
{split_table}

- **train**: para entrenar el modelo. Con varias semillas, cada partido aparece con varias
  degradaciones distintas (aumento de datos).
- **val**: para elegir señales, hiperparámetros y umbrales del decodificador.
- **test**: no se toca hasta tener un candidato final; da una estimación sin sesgo.
- **comprobación** (no está aquí): los 5 minutos públicos del partido del reto, que se ejecutan
  en local y sirven solo para ver que el pipeline funciona. Nunca para elegir.

## Qué hay en cada partido
```
matches/<PARTIDO>/meta.json              equipos, partes (frames), campo, estadísticas de la verdad
matches/<PARTIDO>/labels.csv             frame,label (H/A/D/X): el formato del scorer oficial
matches/<PARTIDO>/truth.npz              tracking limpio (para experimentos de oráculo)
matches/<PARTIDO>/seed<k>/tracking.npz   tracking degradado tipo ATD: 46 slots + balón
matches/<PARTIDO>/seed<k>/smart_tagging.xml   Smart Tagging sintético (formato SportsCode)
matches/<PARTIDO>/seed<k>/degradation.json    parámetros y estadísticas de la degradación
```
Posiciones en centímetros (int16) desde la esquina superior izquierda (x a lo largo del campo,
y hacia abajo), con -32768 = sin dato. Es lo mismo que las coordenadas 0–1 del ATD multiplicadas
por el tamaño del campo. Cárgalo con `load_match()` de `tools/build_datasets.py`.

La línea de tiempo de cada partido: {pad_before} s sin juego, la 1ª parte, {halftime} s de
descanso, la 2ª parte y {pad_after} s sin juego. Fuera de las partes la verdad es X y no se puntúa.

## En qué se parece y en qué no al partido del reto
Parecido, por construcción:
- 25 fps, 46 slots anónimos (23 por equipo), sin identidades.
- Balón ausente en ~41% de los frames evaluados, sobre todo con mucha gente cerca, con el balón
  en el aire o muy rápido y fuera del campo (FAQ.md).
- Jugadores que se pierden y vuelven en otro slot, intercambios entre compañeros (y alguna vez
  entre rivales), duplicados y ruido; unos 20,5 objetos de media (DATA.md).
- Los equipos cambian de lado en el descanso, como en la cámara real.
- Smart Tagging solo con SET PIECES, con la misma proporción por tipo que el archivo real, saques
  de banda con `Team = N/A`, algún equipo equivocado y algún tag falso.

Distinto, y conviene recordarlo:
- La degradación es un modelo nuestro, no el comportamiento real del tracker de Metrica.
- No hay fases tácticas en el Smart Tagging sintético (BUILD UP, PROGRESSION…): no sabemos con
  qué calidad las etiqueta Metrica, y simularlas metería supuestos nuestros en el modelo.
- No hay vídeo ni homografía.
- La definición de posesión de la DFL puede no coincidir al 100% con la del tracking corregido
  de Metrica. Compara `ground_truth` de cada `meta.json` con los 5 minutos públicos
  (17 posesiones de 14,2 s de media) solo para entender la definición, no para ajustar nada.

## Estadísticas
{stats_table}
{verify_text}
"""


def write_readme(out_dir, metas, rows, splits, seeds_per_split, verify_result):
    split_of = {m: s for s, items in splits.items() for m, _ in items}
    split_table = "| partición | partidos | semillas |\n|---|---|---|\n" + "\n".join(
        "| %s | %s | %s |" % (s, ", ".join(sorted({m for m, _ in items})), seeds_per_split[s])
        for s, items in splits.items())
    stats = ["| partido | partición | división | min evaluados | posesiones | posesiones/min | duración media | % D | balón visible | objetos |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for meta in metas:
        gt = meta["ground_truth"]
        r = [x for x in rows if x["match"] == meta["match"]]
        stats.append("| %s (%s) | %s | %s | %.0f | %d | %.2f | %.1f s | %.1f%% | %.1f%% | %.1f |" % (
            meta["match"], meta["title"], split_of.get(meta["match"], "-"), meta["division"], gt["graded_minutes"],
            gt["possessions"], gt["possessions_per_min"], gt["mean_possession_s"], 100 * gt["share"]["D"],
            100 * np.mean([x["ball_visible_graded"] for x in r]), np.mean([x["mean_tracked_objects_graded"] for x in r])))
    stats.append("| *referencia: 5 min públicos del reto* | | | 5 | 17 | 3,40 | 14,2 s | 19,6% | 59,3%* | 20,5* |")
    stats.append("\n\\* del partido completo, según DATA.md.")
    verify_text = ""
    if verify_result:
        table, mean, same = verify_result
        verify_text = ("\n## Comprobación\nBaselines oficiales (más cercano + voto por mayoría) puntuados con el scorer "
                       "oficial, media de todos los partidos: raw %.3f · smooth25 %.3f · smooth50 %.3f · smooth125 %.3f. "
                       "En el partido real el orden es smooth50 > smooth125 > smooth25 > raw: aquí %s.\n" % (
                           mean[0], mean[25], mean[50], mean[125],
                           "**se repite**, buena señal de que estos datos ordenan los métodos como el partido real" if same else
                           "**no se repite**: tratad con cautela las conclusiones sacadas de estos datos"))
    seeds = ", ".join("%s ×%d" % (k, v) for k, v in seeds_per_split.items())
    text = README.format(date=time.strftime("%Y-%m-%d"), seeds=seeds, citation=CITATION,
                         split_table=split_table, pad_before=PAD_BEFORE_S, halftime=HALFTIME_S, pad_after=PAD_AFTER_S,
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
    parser.add_argument("--out", default=None, help="where to write the datasets (default: external/datasets)")
    parser.add_argument("--matches", nargs="*", default=None, help="subset of match ids (default: all seven)")
    parser.add_argument("--train-seeds", type=int, default=2, help="degradations per training match")
    parser.add_argument("--eval-seeds", type=int, default=1, help="degradations per val/test match")
    parser.add_argument("--verify", action="store_true", help="score the official baselines on every match")
    parser.add_argument("--selftest", action="store_true", help="build from fake_data/seed0 instead (no download)")
    args = parser.parse_args()

    prm = dict(DEGRADE)
    t0 = time.time()
    if args.selftest:
        out_dir = os.path.abspath(args.out or os.path.join(ROOT, "fake_data", "selftest_datasets"))
        print("self-test: fake_data/seed0 -> %s" % out_dir)
        clean = clean_from_fake(os.path.join(ROOT, "fake_data", "seed0", "truth.npz"))
        meta, rows = write_match(clean, out_dir, [0, 1], prm)
        metas, all_rows = [meta], [dict(r, match="FAKE0") for r in rows]
        splits = {"train": [["FAKE0", 0]], "val": [["FAKE0", 1]], "test": []}
        seeds_per_split = {"train": 1, "val": 1, "test": 0}
    else:
        out_dir = os.path.abspath(args.out or os.path.join(ROOT, "external", "datasets"))
        matches = args.matches or list(IDSSE)
        splits, metas, all_rows = {"train": [], "val": [], "test": []}, [], []
        seeds_per_split = {"train": args.train_seeds, "val": args.eval_seeds, "test": args.eval_seeds}
        for match in matches:
            split = next(s for s, ms in DEFAULT_SPLITS.items() if match in ms)
            seeds = list(range(seeds_per_split[split]))
            print("%s  %s  (%s)" % (match, IDSSE[match][0], split))
            meta_path, pos_path = fetch_idsse(match, os.path.abspath(args.raw_dir))
            t1 = time.time()
            clean = parse_dfl(meta_path, pos_path, match)
            gt = possession_stats(clean["labels"], clean["halves"])
            print("    parsed in %.0f s: %.0f min, %d possessions (%.2f/min, mean %.1f s), D %.1f%%" % (
                time.time() - t1, gt["graded_minutes"], gt["possessions"], gt["possessions_per_min"],
                gt["mean_possession_s"], 100 * gt["share"]["D"]))
            meta, rows = write_match(clean, out_dir, seeds, prm)
            metas.append(meta)
            all_rows += [dict(r, match=match) for r in rows]
            splits[split] += [[match, s] for s in seeds]
            del clean
    with open(os.path.join(out_dir, "splits.json"), "w") as handle:
        json.dump(splits, handle, indent=2)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["match", "seed", "possessions", "possessions_per_min", "mean_possession_s", "dead_share",
                         "ball_visible_graded", "mean_tracked_objects_graded", "smart_tagging_instances"])
        for r in all_rows:
            gt = next(m for m in metas if m["match"] == r["match"])["ground_truth"]
            writer.writerow([r["match"], r["seed"], gt["possessions"], gt["possessions_per_min"], gt["mean_possession_s"],
                             gt["share"]["D"], r["ball_visible_graded"], r["mean_tracked_objects_graded"],
                             r["smart_tagging_instances"]])
    result = verify(out_dir, {s: v for s, v in splits.items() if v}) if args.verify else None
    write_readme(out_dir, metas, all_rows, splits, seeds_per_split, result)
    print("\ndatasets in %s  (%.0f s)" % (out_dir, time.time() - t0))


if __name__ == "__main__":
    main()
