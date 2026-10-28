#!/usr/bin/env python3
"""Generate a synthetic match in exactly the format of the More than a Hack 2027 data bundle.

Why this exists
---------------
The real match data may not be shown to any cloud-hosted LLM (FAQ.md and the terms of use), and
the public 5 minutes may not be used to choose between versions of a method (RULES.md). So the
team develops against a FAKE match that has the same files, the same formats and the same frame
layout as the real one -- and, unlike the real one, a COMPLETE ground truth, because we made it.

Put this file in tools/ inside a clone of the official repository (it imports games.py), then:

    python tools/make_fake_data.py --seed 0                    # ~1-2 minutes, ~200 MB
    MTAH_DATA_DIR=fake_data/seed0 python starter/baseline.py --out fake_data/seed0/results
    python evaluation/run_evaluation.py --gt fake_data/seed0/BAR-ATH_possession_gt_fake.csv \
        --results fake_data/seed0/results --out fake_data/seed0/eval

Everything that reads the real bundle through games.asset_paths() reads this one unchanged.

What it writes (under --out, default fake_data/seed<SEED>)
    tactical/BAR-ATH_tactical_atd.xml             FIFA/EPTS metadata, 46 track slots, loads in kloppy
    tactical/BAR-ATH_tactical_atd.txt             199,319 lines "frame_id:slot;slot;...:ball"
    tactical/BAR-ATH_tactical_homography.csv      one fixed camera, valid on every frame
    tactical/BAR-ATH_tactical_smart_tagging.xml   automatic-style phases and set pieces, with errors
    BAR-ATH_possession_gt_fake.csv                frame,label for all 199,319 frames, X outside halves
    truth.npz                                     the clean simulation, for testing single modules
    fake_meta.json                                seed, parameters and summary statistics
No video: there is nothing to decode.

How it works, in two stages
---------------------------
1. SIMULATE. A small state machine plays a football match frame by frame at 25 fps, on the real
   match's timeline (same halves, same kick-off frames, the second half graded from 110,808 but
   kicked off at 113,742). The ball is HELD (a carrier dribbles), in a PASS, LOOSE (a duel, a
   deflection, an aerial ball) or DEAD (out of play, foul, goal, waiting for a restart).
   Twenty-two players follow a 4-3-3 shape that shifts with the ball, the nearest defender presses,
   receivers run to the ball. Possessions end the ways real ones do: interceptions, tackles, balls
   out for throw-ins, goal kicks and corners, fouls, shots, saves and goals.
   The ground truth follows the challenge's rules (CHALLENGE.md): possession changes only when the
   other team gains CONTROL; a deflection or a lost aerial duel does not change it; dead ball is D.

2. DEGRADE, so the files look like automatic tracking rather than a clean simulation:
   - the ball is removed on ~41% of graded frames, preferring crowded moments, balls in the air,
     fast balls and balls off the pitch, in runs rather than isolated frames;
   - now and then the "ball" is detected somewhere wrong for a fraction of a second;
   - players are lost for short gaps (more in crowds) and come back in a NEW track slot;
   - nearby team-mates swap slots; rarely, opponents swap too, which is a team error;
   - tracks fragment, and a few ghost duplicates appear;
   - positions get noise; speeds are computed from the noisy positions, like a tracker's.
   The Smart Tagging is derived from the simulated events with missed tags, false ones, wrong teams,
   blurred boundaries and Team = N/A on every throw-in, as in the real file.

What it is NOT
--------------
It is not a validation set for choosing between methods. The simulator is a set of assumptions:
a method tuned on it learns the simulator. Use it to build and test code, to check that a module
does what you think, and to stress-test robustness (--ball-missing, --noise-scale). Choose between
methods on real tracking from other matches (RULES.md allows that), never on this.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import games  # noqa: E402

# --- the real match's layout, read from games.py so nothing drifts ---------------------------------
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
DT = 1.0 / FPS
N_FRAMES = games.source_frames("tactical", GAME)
L, W = games.GAMES[GAME]["pitch"]                     # metres
HALF_BLOCKS = games.half_video_blocks(GAME)           # [(1, 11185, 90651), (2, 110808, 190197)]
KICKOFF = {1: 11185, 2: 113742}                       # published in DATA.md and FAQ.md
CENTRE = np.array([L / 2.0, W / 2.0])

HOME, AWAY = 0, 1
TEAM = np.repeat([HOME, AWAY], 11)                    # player index -> team
GK = (0, 11)                                          # goalkeeper index per team
SLOTS_PER_TEAM = 23
N_SLOTS = 2 * SLOTS_PER_TEAM                          # 46, as in the real ATD
ATD_TEAM_NAMES = ("Barcelona", "Athletic Club")       # the real ATD names may differ: map by ORDER
TAG_TEAM_NAMES = ("Barcelona", "Atletic Club")        # exactly as in the real Smart Tagging
GOAL_HALF_WIDTH = 3.66
BOX_DEPTH = (L - 16.5) / L                            # depth (0..1 towards the opponent) of the box
BOX_HALF_WIDTH = 20.16

# 4-3-3 as (depth, lateral): depth 0 = own goal line, 1 = opponent's goal line; lateral 0..1 across.
FORMATION = np.array([
    (0.03, 0.50),
    (0.20, 0.14), (0.17, 0.38), (0.17, 0.62), (0.20, 0.86),
    (0.37, 0.25), (0.33, 0.50), (0.37, 0.75),
    (0.56, 0.14), (0.60, 0.50), (0.56, 0.86),
])

# Seconds the ball stays dead before each kind of restart (uniform between the two values).
DEAD_SECONDS = {"throw_in": (6, 18), "goal_kick": (12, 28), "corner": (18, 40),
                "free_kick": (12, 45), "kickoff_goal": (55, 95)}

DEFAULTS = {
    "ball_missing": 0.41,      # share of graded frames without a ball (DATA.md: 41%)
    "ball_noise_m": 0.25,      # positional noise on the ball
    "ball_false_rate": 0.01,   # share of visible frames covered by a wrong ball detection
    "player_noise_m": 0.12,    # positional noise on players
    "player_loss": 0.0025,     # per-frame chance of losing a player (x up to 2.5 in crowds)
    "player_gap_s": 1.2,       # mean length of a lost-player gap
    "swap_same_team": 0.02,    # per-frame chance two close team-mates swap slots
    "swap_cross_team": 0.004,  # per-frame chance two close opponents swap slots (team error)
    "fragment": 0.0004,        # per-frame chance a track jumps to a new slot with no gap
    "duplicate": 0.0002,       # per-frame chance of a short ghost duplicate
    "tag_recall": 0.65,         # share of true phases that get a Smart Tagging instance
    "tag_false": 0.04,         # false Smart Tagging instances, as a share of true ones
}


def rotate(vec, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def unit(vec):
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-9 else np.array([1.0, 0.0])


# ==================================================================================================
# Stage 1 -- simulate a match
# ==================================================================================================

class MatchSim:
    """Plays the match frame by frame and records the clean truth.

    State is one of HOLD, PASS, LOOSE, DEAD. `pt` is the team in possession: it only changes when
    a player of the other team gains control (gain_control) or at a restart (do_restart), which is
    the challenge's definition. Everything is in metres, origin at the top-left corner.
    """

    def __init__(self, rng, home_right_first):
        self.rng = rng
        self.home_right_first = home_right_first
        n = N_FRAMES
        self.true_pos = np.full((n, 22, 2), np.nan, dtype=np.float32)
        self.true_ball = np.full((n, 2), np.nan, dtype=np.float32)
        self.ball_air = np.zeros(n, dtype=bool)
        self.carrier_track = np.full(n, -1, dtype=np.int8)
        self.gt = np.full(n, "X", dtype="<U1")
        self.graded = np.zeros(n, dtype=bool)
        for _, first, last in HALF_BLOCKS:
            self.graded[first:last + 1] = True

        self.spells, self.restarts, self.goals, self.shots, self.long_balls = [], [], [], [], []
        self.p = np.zeros((22, 2))
        self.v = np.zeros((22, 2))
        self.wander = np.zeros((22, 2))
        self.ball_xy = CENTRE.copy()
        self.ball_vel = np.zeros(2)
        self.state, self.half, self.pt = "DEAD", 1, HOME
        self.spell, self.restart, self.carrier, self.last_touch = None, None, -1, HOME
        self.overrides = {}
        self.pass_air = False

    # --- geometry --------------------------------------------------------------------------------
    def sign(self, team):
        """+1 if `team` attacks towards x = L in the current half, else -1."""
        s = 1.0 if self.home_right_first else -1.0
        if self.half == 2:
            s = -s
        return s if team == HOME else -s

    def depth(self, team, x):
        return x / L if self.sign(team) > 0 else 1.0 - x / L

    def x_at(self, team, d):
        return d * L if self.sign(team) > 0 else (1.0 - d) * L

    @staticmethod
    def players_of(team):
        return np.arange(11) if team == HOME else np.arange(11, 22)

    def nearest(self, team, point, k=1, outfield=False, exclude=()):
        idx = self.players_of(team)
        if outfield:
            idx = idx[1:]
        if exclude:
            idx = np.array([i for i in idx if i not in exclude])
        order = idx[np.argsort(np.linalg.norm(self.p[idx] - point, axis=1))]
        return order[:k] if k > 1 else int(order[0])

    # --- movement --------------------------------------------------------------------------------
    def team_targets(self, team, ball, in_poss, kickoff=False):
        """Where each player of `team` wants to be: the formation, shifted and squeezed by the ball."""
        bd = float(np.clip(self.depth(team, ball[0]), 0.0, 1.0))
        bl = float(np.clip(ball[1] / W, 0.0, 1.0))
        d, l = FORMATION[:, 0].copy(), FORMATION[:, 1].copy()
        scale_d, scale_l, bias = (1.0, 1.0, 0.07) if in_poss else (0.72, 0.80, -0.06)
        d[1:] = 0.37 + (d[1:] - 0.37) * scale_d + (bd - 0.5) * 0.55 + bias
        l[1:] = 0.5 + (l[1:] - 0.5) * scale_l + (bl - 0.5) * 0.30
        d[0], l[0] = 0.025 + 0.07 * bd, 0.5 + (bl - 0.5) * 0.25
        if kickoff:
            d[1:] = np.minimum(d[1:], 0.46)
        d, l = np.clip(d, 0.01, 0.99), np.clip(l, 0.03, 0.97)
        wander = self.wander[self.players_of(team)].copy()
        wander[0] *= 0.2
        return np.stack([self.x_at(team, d), l * W], axis=1) + wander

    def both_targets(self, ball, team_in_poss, kickoff=False):
        return np.concatenate([self.team_targets(HOME, ball, team_in_poss == HOME, kickoff),
                               self.team_targets(AWAY, ball, team_in_poss == AWAY, kickoff)])

    def move(self, targets, vmax):
        desired = (targets - self.p) / 0.7
        speed = np.linalg.norm(desired, axis=1)
        desired *= np.minimum(1.0, vmax / np.maximum(speed, 1e-6))[:, None]
        self.v += (desired - self.v) * 0.12
        self.p += self.v * DT
        np.clip(self.p[:, 0], -2.0, L + 2.0, out=self.p[:, 0])
        np.clip(self.p[:, 1], -2.0, W + 2.0, out=self.p[:, 1])

    def wander_step(self):
        a = math.exp(-DT / 6.0)
        self.wander = a * self.wander + math.sqrt(1 - a * a) * 3.5 * self.rng.normal(size=(22, 2))

    # --- possession bookkeeping ------------------------------------------------------------------
    def start_spell(self, f, team, cause):
        self.close_spell(f)
        self.pt = team
        bd = float(self.depth(team, self.ball_xy[0]))
        self.spell = {"team": team, "half": self.half, "start": f, "end": None, "cause": cause,
                      "start_depth": bd, "max_depth": bd, "final_third": None, "in_box": None,
                      "box_players": 0}
        self.budget_end = f + int(self.rng.lognormal(math.log(9.0), 0.75) * FPS)

    def close_spell(self, f):
        if self.spell is not None:
            self.spell["end"] = f - 1
            if self.spell["end"] >= self.spell["start"]:
                self.spells.append(self.spell)
            self.spell = None

    def gain_control(self, f, j):
        """Player j controls the ball. A change of possession only if j is on the other team."""
        team = int(TEAM[j])
        if self.spell is None or team != self.pt:
            self.start_spell(f, team, "turnover" if self.spell is not None else "restart")
        self.carrier, self.last_touch, self.state = j, team, "HOLD"
        self.hold_left = int(self.rng.uniform(0.6, 2.2) * FPS)
        self.drib_dir = rotate(np.array([self.sign(team), 0.0]), self.rng.normal(0, math.radians(45)))
        self.drib_speed = self.rng.uniform(1.5, 4.5)
        self.ball_xy = self.p[j] + 0.5 * self.drib_dir

    # --- actions ---------------------------------------------------------------------------------
    def fly(self, f, target, speed, kind, receiver, air=False):
        self.pass_from = self.ball_xy.copy()
        self.pass_to = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(self.pass_to - self.pass_from))
        self.pass_start, self.pass_end = f, f + max(6, int(round(dist / speed * FPS)))
        self.pass_kind, self.receiver, self.pass_air = kind, receiver, air
        self.state, self.carrier = "PASS", -1

    def interceptor(self, start, end, team):
        """The player of `team` best placed to cut the line start->end, and where."""
        idx = self.players_of(team)[1:]
        seg = end - start
        t = np.clip(((self.p[idx] - start) @ seg) / max(seg @ seg, 1e-9), 0.3, 0.9)
        points = start + t[:, None] * seg[None, :]
        k = int(np.argmin(np.linalg.norm(self.p[idx] - points, axis=1)))
        return int(idx[k]), points[k] + self.rng.normal(0, 0.5, 2)

    def pass_ball(self, f, intercept=False, deflect=False, max_len=None, short=False):
        rng, team, bp = self.rng, self.pt, self.ball_xy
        mates = np.array([i for i in self.players_of(team) if i != self.carrier])
        dist = np.linalg.norm(self.p[mates] - bp, axis=1)
        fwd = self.depth(team, self.p[mates, 0]) - self.depth(team, bp[0])
        opp = self.p[self.players_of(1 - team)]
        openness = np.linalg.norm(self.p[mates][:, None, :] - opp[None, :, :], axis=2).min(axis=1)
        if short:
            w = np.exp(-((dist - 9.0) / 5.0) ** 2)
        else:
            w = np.exp(-((dist - 17.0) / 11.0) ** 2) * (1 + 1.2 * np.clip(fwd * 4, -0.5, 1.0))
        w = w * np.clip(openness / 4.0, 0.2, 1.5)
        if max_len:
            w[dist > max_len] *= 0.02
        w[mates == GK[team]] *= 0.4 if self.depth(team, bp[0]) < 0.3 else 0.02
        w = np.maximum(w, 1e-9)
        r = int(mates[rng.choice(len(mates), p=w / w.sum())])
        speed = rng.uniform(12.0, 17.0)
        lead = float(np.linalg.norm(self.p[r] - bp)) / speed
        target = self.p[r] + self.v[r] * lead * 0.8
        target = np.array([np.clip(target[0], 1, L - 1), np.clip(target[1], 1, W - 1)])
        air = float(np.linalg.norm(target - bp)) > 32
        self.last_touch = team
        if intercept or deflect:
            j, point = self.interceptor(bp.copy(), target, 1 - team)
            self.fly(f, point, speed, "intercept" if intercept else "deflect", j, air)
        else:
            self.fly(f, target, speed, "pass", r, air)

    def long_ball(self, f, lose=False, cross=False):
        rng, team, bp = self.rng, self.pt, self.ball_xy
        s = self.sign(team)
        if cross:
            target = np.array([self.x_at(team, rng.uniform(0.88, 0.96)), W / 2 + rng.normal(0, 7)])
            keep = 0.35
        else:
            mates = self.players_of(team)[1:]
            mates = mates[mates != self.carrier]
            ahead = mates[self.depth(team, self.p[mates, 0]) > self.depth(team, bp[0]) + 0.15]
            if len(ahead) == 0:
                return self.pass_ball(f)
            ahead = ahead[np.argsort(-self.depth(team, self.p[ahead, 0]))][:3]
            r = int(rng.choice(ahead))
            target = self.p[r] + np.array([s * 3.0, 0.0])
            keep = 0.55
        target = np.array([np.clip(target[0], 1, L - 1), np.clip(target[1], 1, W - 1)])
        if not cross and np.linalg.norm(target - bp) < 25:
            return self.pass_ball(f)
        self.aerial_keep = 0.0 if lose else keep
        self.long_balls.append({"frame": f, "team": team, "half": self.half})
        self.last_touch = team
        self.fly(f, target, 20.0, "aerial", None, air=True)

    def shoot(self, f):
        rng, team = self.rng, self.pt
        s, goal_x, keeper = self.sign(team), self.x_at(team, 1.0), GK[1 - team]
        r = rng.random()
        self.shots.append({"frame": f, "team": team, "half": self.half})
        if r < 0.09:
            target, kind, recv = (goal_x + s * 1.5, W / 2 + rng.uniform(-3.2, 3.2)), "shot_goal", None
        elif r < 0.53:
            target = self.p[keeper] + np.array([-s * 0.8, 0.0])
            kind, recv = ("save" if r < 0.41 else "save_corner"), keeper
        else:
            side = rng.choice([-1.0, 1.0])
            target = (goal_x + s * 2.0, W / 2 + side * (GOAL_HALF_WIDTH + rng.uniform(0.5, 9.0)))
            kind, recv = "shot_wide", None
        self.last_touch = team
        self.fly(f, target, 24.0, kind, recv, air=rng.random() < 0.4)

    def out_pass(self, f, line):
        rng, team, bp = self.rng, self.pt, self.ball_xy
        s = self.sign(team)
        if line == "touch":
            target = (np.clip(bp[0] + s * rng.uniform(5, 25), 2, L - 2), -2.0 if bp[1] < W / 2 else W + 2.0)
        else:
            y = rng.uniform(2, W / 2 - 5) if rng.random() < 0.5 else rng.uniform(W / 2 + 5, W - 2)
            target = (self.x_at(team, 1.0) + s * 2.0, y)
        self.fly(f, target, 15.0, "out", None)
        deflected = 0.3 if line == "touch" else 0.45                 # deflected out by them
        self.last_touch = 1 - team if rng.random() < deflected else team

    def loose(self, f, winner, speed=None, seconds=(0.4, 1.0), direction=None):
        if speed is not None:
            d = direction if direction is not None else rotate(np.array([1.0, 0.0]),
                                                               self.rng.uniform(0, 2 * math.pi))
            self.ball_vel = unit(d) * speed
        self.loose_winner = winner
        self.loose_end = f + int(self.rng.uniform(*seconds) * FPS)
        self.state, self.carrier = "LOOSE", -1

    def contest(self, f, winner):
        """A duel at the carrier's feet. Possession changes only if the other team wins it."""
        self.last_touch = int(self.rng.integers(2))
        self.loose(f, winner, speed=self.rng.uniform(2.0, 4.0), seconds=(0.3, 1.0))

    def foul(self, f):
        rng, team = self.rng, self.pt
        fk_team = team if rng.random() < 0.75 else 1 - team
        spot = np.array([np.clip(self.ball_xy[0], 1, L - 1), np.clip(self.ball_xy[1], 1, W - 1)])
        attack = 1 - fk_team if fk_team != team else team
        if self.depth(attack, spot[0]) >= BOX_DEPTH - 0.01 and abs(spot[1] - W / 2) <= BOX_HALF_WIDTH:
            spot[0] = self.x_at(attack, BOX_DEPTH - 0.02)                # no penalties in this sim
        self.enter_dead(f, "free_kick", fk_team, spot)

    def decide(self, f):
        """The carrier's hold is over: keep the possession going, or end it if its time is up."""
        rng, team = self.rng, self.pt
        bd = self.depth(team, self.ball_xy[0])
        if f < self.budget_end:
            r = rng.random()
            if r < 0.06:
                self.contest(f, winner=team)
            elif r < 0.14 and bd < 0.55:
                self.long_ball(f)
            else:
                self.pass_ball(f, deflect=rng.random() < 0.07)
            return
        if bd > 0.7:
            options = (("shot", .22), ("intercept", .22), ("tackle", .22), ("out_goal", .16),
                       ("foul", .10), ("out_touch", .08))
        else:
            options = (("intercept", .33), ("tackle", .27), ("out_touch", .22), ("foul", .12),
                       ("long_lost", .06))
        names, probs = zip(*options)
        choice = names[rng.choice(len(names), p=np.array(probs) / sum(probs))]
        if choice == "shot":
            self.shoot(f)
        elif choice == "intercept":
            self.pass_ball(f, intercept=True)
        elif choice == "tackle":
            self.contest(f, winner=1 - team)
        elif choice == "out_goal":
            self.out_pass(f, "goal")
        elif choice == "out_touch":
            self.out_pass(f, "touch")
        elif choice == "foul":
            self.foul(f)
        else:
            self.long_ball(f, lose=True)

    def arrive(self, f):
        rng, kind = self.rng, self.pass_kind
        if kind in ("pass", "intercept", "save"):
            self.gain_control(f, self.receiver)
        elif kind == "deflect":
            self.last_touch = int(TEAM[self.receiver])
            direction = rotate(unit(self.pass_to - self.pass_from), rng.normal(0, math.radians(60)))
            self.loose(f, self.pt, speed=rng.uniform(5, 9), direction=direction)
        elif kind == "aerial":
            winner = self.pt if rng.random() < self.aerial_keep else 1 - self.pt
            self.last_touch = int(rng.integers(2))
            self.loose(f, winner, speed=rng.uniform(3, 7))
        elif kind == "save_corner":
            defending = int(TEAM[self.receiver])
            self.last_touch = defending
            self.loose(f, defending, speed=18.0, seconds=(3.0, 3.0),
                       direction=np.array([self.sign(self.pt), rng.choice([-0.45, 0.45])]))

    def resolve_loose(self, f):
        j = self.nearest(self.loose_winner, self.ball_xy)
        if np.linalg.norm(self.p[j] - self.ball_xy) > 1.5 and f < self.loose_end + int(1.5 * FPS):
            return                                                        # still chasing it
        self.gain_control(f, j)

    # --- dead ball -------------------------------------------------------------------------------
    def enter_dead(self, f, kind, team, spot, frame=None):
        rng = self.rng
        self.close_spell(f)
        self.state, self.carrier, self.overrides = "DEAD", -1, {}
        spot = np.asarray(spot, dtype=float)
        if frame is None and kind != "end_whistle":
            lo, hi = DEAD_SECONDS[kind]
            frame = f + int(rng.uniform(lo, hi) * FPS)
        self.restart = {"type": kind, "team": team, "spot": spot, "frame": frame, "taker": None}
        if kind == "end_whistle":
            return
        base = 0 if team == HOME else 11
        if kind == "goal_kick":
            taker = GK[team]
        elif kind.startswith("kickoff"):
            taker = base + 9
            self.overrides[base + 8] = spot + np.array([0.0, 1.5])
        else:
            taker = self.nearest(team, spot, outfield=True)
        self.restart["taker"] = taker
        self.overrides[taker] = spot.copy()
        if kind == "corner":
            attackers = [i for i in self.players_of(team)[1:] if i != taker]
            attackers = sorted(attackers, key=lambda i: -self.depth(team, self.p[i, 0]))[:5]
            for i in attackers:
                self.overrides[i] = np.array([self.x_at(team, rng.uniform(0.88, 0.97)),
                                              W / 2 + rng.uniform(-9, 9)])
            for i in self.players_of(1 - team)[1:8]:
                self.overrides[i] = np.array([self.x_at(team, rng.uniform(0.90, 0.98)),
                                              W / 2 + rng.uniform(-10, 10)])
            self.overrides[GK[1 - team]] = np.array([self.x_at(team, 0.995), W / 2])

    def do_restart(self, f):
        rng, r = self.rng, self.restart
        team, taker, kind = r["team"], r["taker"], r["type"]
        self.p[taker], self.v[taker] = r["spot"], 0.0
        self.ball_xy, self.ball_vel = r["spot"].copy(), np.zeros(2)
        self.restarts.append({"type": kind, "team": team, "frame": f, "half": self.half})
        self.start_spell(f, team, kind)
        self.carrier, self.last_touch, self.overrides = taker, team, {}
        if kind.startswith("kickoff"):
            self.pass_ball(f, short=True)
        elif kind == "throw_in":
            self.pass_ball(f, max_len=20)
        elif kind == "goal_kick":
            self.long_ball(f) if rng.random() < 0.6 else self.pass_ball(f, short=True)
        elif kind == "corner":
            self.long_ball(f, cross=True)
        elif self.depth(team, r["spot"][0]) > 0.65 and rng.random() < 0.6:
            self.long_ball(f, cross=True)
        else:
            self.pass_ball(f)

    def check_out(self, f):
        """The ball has crossed a line: throw-in, goal kick, corner or goal."""
        x, y = self.ball_xy
        if 0 <= x <= L and 0 <= y <= W:
            return
        if self.state == "PASS":
            span = max(1, self.pass_end - self.pass_start)
            self.ball_vel = (self.pass_to - self.pass_from) / span * FPS * 0.5
        if 0 <= x <= L:
            spot = np.array([np.clip(x, 1, L - 1), 0.0 if y < 0 else W])
            return self.enter_dead(f, "throw_in", 1 - self.last_touch, spot)
        at_right = x > L
        defending = HOME if (self.sign(HOME) < 0) == at_right else AWAY
        attacking = 1 - defending
        if self.state == "PASS" and self.pass_kind == "shot_goal" and abs(y - W / 2) <= GOAL_HALF_WIDTH:
            self.goals.append({"frame": f, "team": attacking, "half": self.half})
            return self.enter_dead(f, "kickoff_goal", defending, CENTRE.copy())
        if self.last_touch == defending:
            spot = np.array([L if at_right else 0.0, 0.0 if y < W / 2 else W])
            return self.enter_dead(f, "corner", attacking, spot)
        spot = np.array([self.x_at(defending, 0.05), W / 2 + self.rng.uniform(-9, 9)])
        return self.enter_dead(f, "goal_kick", defending, spot)

    # --- one frame -------------------------------------------------------------------------------
    def transitions(self, f):
        st = self.state
        if st == "DEAD":
            if self.restart["frame"] is not None and f >= self.restart["frame"]:
                self.do_restart(f)
        elif st == "HOLD":
            if self.hold_left <= 0:
                self.decide(f)
        elif st == "PASS":
            if f >= self.pass_end and self.pass_kind not in ("out", "shot_goal", "shot_wide"):
                self.arrive(f)
        elif st == "LOOSE" and f >= self.loose_end:
            self.resolve_loose(f)

    def step_dead(self, f):
        r = self.restart
        if r["frame"] is not None and r["frame"] - f <= 2 * FPS:
            self.ball_xy, self.ball_vel = r["spot"].copy(), np.zeros(2)       # ball placed
        else:
            self.ball_xy = self.ball_xy + self.ball_vel * DT
            self.ball_vel *= 0.92
        anchor = self.ball_xy if r["type"] == "end_whistle" else r["spot"]
        targets = self.both_targets(anchor, r["team"], kickoff=r["type"].startswith("kickoff"))
        vmax = np.full(22, 3.0)
        for i, t in self.overrides.items():
            targets[i], vmax[i] = t, 6.0
        self.move(targets, vmax)

    def step_hold(self, f):
        c, team = self.carrier, self.pt
        pos, dvec, s = self.p[c].copy(), self.drib_dir.copy(), self.sign(team)
        if pos[1] < 5:
            dvec[1] += 0.15
        if pos[1] > W - 5:
            dvec[1] -= 0.15
        dd = self.depth(team, pos[0])
        if dd > 0.93:
            dvec[0] -= 0.2 * s
        if dd < 0.05:
            dvec[0] += 0.2 * s
        self.drib_dir = dvec = unit(dvec)
        targets = self.both_targets(self.ball_xy, team)
        vmax = np.full(22, 6.5)
        presser, cover = self.nearest(1 - team, self.ball_xy, k=2, outfield=True)
        back = np.array([s, 0.0])            # the defenders' own goal is behind them, along +s
        targets[presser], vmax[presser] = self.ball_xy + back * 2.5, 7.5
        targets[cover] = self.ball_xy + back * 8.0
        self.move(targets, vmax)
        self.p[c] = pos + dvec * self.drib_speed * DT
        self.v[c] = dvec * self.drib_speed
        self.ball_xy = self.p[c] + 0.6 * dvec
        self.hold_left -= 1

    def step_pass(self, f):
        t = (f - self.pass_start) / max(1, self.pass_end - self.pass_start)
        self.ball_xy = self.pass_from + (self.pass_to - self.pass_from) * t
        targets = self.both_targets(self.ball_xy, self.pt)
        vmax = np.full(22, 6.5)
        if self.receiver is not None:
            targets[self.receiver], vmax[self.receiver] = self.pass_to, 8.0
            if self.pass_kind == "pass":
                j = self.nearest(1 - self.pt, self.pass_to, outfield=True)
                targets[j] = self.pass_to + np.array([self.sign(self.pt) * 2.0, 0.0])
                vmax[j] = 7.5
        self.move(targets, vmax)

    def step_loose(self, f):
        self.ball_xy = self.ball_xy + self.ball_vel * DT
        self.ball_vel *= 0.94
        targets = self.both_targets(self.ball_xy, self.pt)
        vmax = np.full(22, 6.5)
        for team in (HOME, AWAY):
            for j in self.nearest(team, self.ball_xy, k=2):
                targets[j], vmax[j] = self.ball_xy, 8.0
        self.move(targets, vmax)

    def record(self, f):
        self.true_pos[f] = self.p
        self.true_ball[f] = self.ball_xy
        self.ball_air[f] = self.state == "PASS" and self.pass_air
        self.carrier_track[f] = self.carrier if self.state == "HOLD" else -1
        if self.graded[f]:
            self.gt[f] = "D" if self.state == "DEAD" else "HA"[self.pt]
        sp = self.spell
        if sp is not None and f % 5 == 0:
            bd = float(self.depth(self.pt, self.ball_xy[0]))
            sp["max_depth"] = max(sp["max_depth"], bd)
            if bd > 2 / 3 and sp["final_third"] is None:
                sp["final_third"] = f
            if bd >= BOX_DEPTH and abs(self.ball_xy[1] - W / 2) <= BOX_HALF_WIDTH and sp["in_box"] is None:
                sp["in_box"] = f
            if bd > 0.7:
                att = self.players_of(self.pt)[1:]
                pd = self.depth(self.pt, self.p[att, 0])
                inside = (pd >= BOX_DEPTH) & (np.abs(self.p[att, 1] - W / 2) <= BOX_HALF_WIDTH)
                sp["box_players"] = max(sp["box_players"], int(inside.sum()))

    def tick(self, f):
        self.wander_step()
        self.transitions(f)
        {"DEAD": self.step_dead, "HOLD": self.step_hold,
         "PASS": self.step_pass, "LOOSE": self.step_loose}[self.state](f)
        if self.state in ("PASS", "LOOSE"):
            self.check_out(f)
        self.record(f)

    def run_half(self, half, first, last, kick_team):
        """Walk on, kick off at the published frame, play, final whistle, walk off."""
        rng = self.rng
        self.half = half
        kickoff = KICKOFF[half]
        appear = kickoff - 60 * FPS if half == 1 else first - 30 * FPS
        self.ball_xy, self.ball_vel = CENTRE.copy(), np.zeros(2)
        self.p = self.both_targets(CENTRE, kick_team, kickoff=True) + rng.normal(0, 10, size=(22, 2))
        self.v[:] = 0.0
        self.spell = None
        self.enter_dead(appear, "kickoff_start", kick_team, CENTRE.copy(), frame=kickoff)
        whistle = last - int(rng.uniform(1, 3) * FPS)
        for f in range(appear, last + 1):
            if f == whistle:
                self.enter_dead(f, "end_whistle", None, self.ball_xy.copy())
            self.tick(f)
        self.close_spell(last + 1)
        exit_point = np.array([L / 2, W + 3.0])
        for f in range(last + 1, min(N_FRAMES, last + 30 * FPS)):          # ungraded walk-off
            self.wander_step()
            self.move(exit_point + 0.5 * self.wander, np.full(22, 1.6))
            self.true_pos[f] = self.p


# ==================================================================================================
# Stage 2 -- degrade it into automatic tracking
# ==================================================================================================

def smooth_noise(rng, n, window):
    noise = rng.normal(size=n + window)
    kernel = np.ones(window) / window
    out = np.convolve(noise, kernel, mode="valid")[:n]
    return (out - out.mean()) / out.std()


def degrade_ball(sim, rng, prm):
    """Hide the ball where automatic tracking loses it, add noise and a few wrong detections."""
    ball = sim.true_ball.astype(np.float64)
    present = ~np.isnan(ball[:, 0])
    graded = sim.graded

    crowd = np.zeros(N_FRAMES)
    for start in range(0, N_FRAMES, 20000):
        sl = slice(start, start + 20000)
        dist = np.linalg.norm(sim.true_pos[sl] - ball[sl, None, :], axis=2)
        crowd[sl] = np.nansum(dist < 4.0, axis=1)
    speed = np.zeros(N_FRAMES)
    speed[1:] = np.nan_to_num(np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS)
    off = present & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))

    dead = sim.gt == "D"
    score = (0.45 * crowd + 1.4 * sim.ball_air + 0.04 * np.minimum(speed, 30) + 3.0 * off
             + 1.0 * dead + smooth_noise(rng, N_FRAMES, 21))
    threshold = np.quantile(score[graded & present], 1.0 - prm["ball_missing"])
    missing = ~present | (score > threshold)
    missing |= present & ~graded & (rng.random(N_FRAMES) < 0.8)

    reported = ball + rng.normal(0, prm["ball_noise_m"], ball.shape)
    visible_graded = np.flatnonzero(graded & ~missing)
    n_runs = int(prm["ball_false_rate"] * len(visible_graded) / 12)
    for start in rng.choice(visible_graded, size=n_runs, replace=False):
        length = int(rng.integers(5, 26))
        offset = rotate(np.array([1.0, 0.0]), rng.uniform(0, 2 * math.pi)) * rng.uniform(6, 30)
        end = min(start + length, N_FRAMES)
        reported[start:end] = ball[start] + offset + rng.normal(0, 0.2, (end - start, 2))
        missing[start:end] = False
    reported[missing] = np.nan
    reported[:, 0] = np.clip(reported[:, 0], -0.05 * L, 1.05 * L)
    reported[:, 1] = np.clip(reported[:, 1], -0.05 * W, 1.05 * W)
    return reported


def degrade_players(sim, rng, prm):
    """Turn 22 clean players into 46 unreliable track slots, the way a tracker would."""
    out = np.full((N_FRAMES, N_SLOTS, 2), np.nan, dtype=np.float32)
    slot_team = np.repeat([HOME, AWAY], SLOTS_PER_TEAM)
    free = [list(range(SLOTS_PER_TEAM)), list(range(SLOTS_PER_TEAM, N_SLOTS))]
    slot_of = np.full(22, -1, dtype=int)
    gap_left = np.zeros(22, dtype=int)
    ghosts = []                                     # [player, slot, offset, frames left]
    same_team = TEAM[:, None] == TEAM[None, :]
    upper = np.triu(np.ones((22, 22), dtype=bool), 1)
    gap_p = 1.0 / (prm["player_gap_s"] * FPS)

    def take(team):
        pool = free[team]
        return pool.pop(int(rng.integers(len(pool)))) if pool else -1

    def release(slot):
        if slot >= 0:
            free[slot_team[slot]].append(slot)

    for f in np.flatnonzero(~np.isnan(sim.true_pos[:, 0, 0])):
        P = sim.true_pos[f].astype(np.float64)
        present = ~np.isnan(P[:, 0])
        for i in np.flatnonzero(~present & (slot_of >= 0)):
            release(slot_of[i])
            slot_of[i] = -1
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        crowd = np.nansum(d < 1.8, axis=1) - 1

        lose = (slot_of >= 0) & (rng.random(22) < prm["player_loss"] * (1 + 1.5 * np.minimum(crowd, 1)))
        for i in np.flatnonzero(lose):
            release(slot_of[i])
            slot_of[i], gap_left[i] = -1, 1 + rng.geometric(gap_p)
        waiting = (slot_of < 0) & present
        gap_left[waiting] -= 1
        for i in np.flatnonzero(waiting & (gap_left <= 0)):
            slot_of[i] = take(TEAM[i])

        for i in np.flatnonzero((slot_of >= 0) & (rng.random(22) < prm["fragment"])):
            new = take(TEAM[i])
            if new >= 0:
                release(slot_of[i])
                slot_of[i] = new

        tracked = (slot_of[:, None] >= 0) & (slot_of[None, :] >= 0) & upper
        for i, j in np.argwhere(tracked & same_team & (d < 1.2)):
            if rng.random() < prm["swap_same_team"]:
                slot_of[i], slot_of[j] = slot_of[j], slot_of[i]
        for i, j in np.argwhere(tracked & ~same_team & (d < 1.0)):
            if rng.random() < prm["swap_cross_team"]:
                slot_of[i], slot_of[j] = slot_of[j], slot_of[i]

        for i in np.flatnonzero((slot_of >= 0) & (rng.random(22) < prm["duplicate"])):
            slot = take(TEAM[i])
            if slot >= 0:
                offset = rotate(np.array([1.0, 0.0]), rng.uniform(0, 2 * math.pi)) * rng.uniform(0.8, 2.0)
                ghosts.append([i, slot, offset, int(rng.integers(10, 41))])

        idx = np.flatnonzero(slot_of >= 0)
        out[f, slot_of[idx]] = P[idx] + rng.normal(0, prm["player_noise_m"], (len(idx), 2))
        alive = []
        for g in ghosts:
            if present[g[0]] and g[3] > 0:
                out[f, g[1]] = P[g[0]] + g[2] + rng.normal(0, 0.2, 2)
                g[3] -= 1
                alive.append(g)
            else:
                release(g[1])
        ghosts = alive
    return out


def speeds(xy, k=3):
    """Speed in m/s from positions (n, ..., 2): central difference over +/-k frames, as a tracker
    would smooth it. NaN where either end is missing, so tracks start and end without a speed."""
    xy = xy.astype(np.float64)
    out = np.full(xy.shape[:-1], np.nan)
    out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) * FPS / (2 * k)
    out[np.isnan(xy[..., 0])] = np.nan
    return out


# ==================================================================================================
# Smart Tagging -- automatic-style phases derived from the simulation, with automatic-style errors
# ==================================================================================================

def build_tags(sim, rng, prm):
    names = TAG_TEAM_NAMES
    half_name = {1: "1st Half", 2: "2nd Half"}
    instances = []

    def add(code, start, end, team, half, **extra):
        labels = {"Team": names[team] if team is not None else "N/A", "Half": half_name[half]}
        labels.update(extra)
        instances.append({"code": code, "start": max(0, start), "end": min(N_FRAMES - 1, end),
                          "labels": labels})

    def sec(lo, hi):
        return int(rng.uniform(lo, hi) * FPS)

    set_piece = {"throw_in": ("Throw-in", 0.55), "corner": ("Corner Kick", 0.9),
                 "goal_kick": ("Goal Kick", 0.55), "free_kick": ("Free Kick", 0.25),
                 "kickoff_goal": ("Kick Off (After Goal)", 1.0),
                 "kickoff_start": ("Kick Off (Start)", 1.0)}
    for r in sim.restarts:
        label, keep = set_piece[r["type"]]
        if rng.random() < keep:
            team = None if r["type"] == "throw_in" else r["team"]   # every throw-in is N/A
            add("SET PIECES", r["frame"] - sec(2, 3), r["frame"] + sec(2, 3), team, r["half"], Type=label)

    for g in sim.goals:
        add("GOALS", g["frame"] - sec(1, 3), g["frame"] + sec(2, 4), g["team"], g["half"])
    for lb in sim.long_balls:
        if rng.random() < 0.6:
            add("LONG BALL", lb["frame"] - sec(0.5, 1.5), lb["frame"] + sec(1.5, 3), lb["team"], lb["half"])

    recall = prm["tag_recall"]
    for sp in sim.spells:
        start, end, half = sp["start"], sp["end"], sp["half"]
        if end - start < 2 * FPS:
            continue
        team = sp["team"] if rng.random() > 0.05 else 1 - sp["team"]          # 5% wrong team
        if rng.random() < 0.015:
            team = None
        other = None if team is None else 1 - team
        jitter_start = start + int(rng.normal(0, 1.0) * FPS)
        jitter_end = end + sec(-2, 3)
        zone = sp["start_depth"]
        if zone < 1 / 3:
            code = "BUILD UP"
        elif zone < 2 / 3:
            code = "PROGRESSION"
        else:
            code = None
        if code and rng.random() < recall:
            add(code, jitter_start, jitter_end, team, half)
        box = sp["box_players"]
        box_label = "7+" if box >= 7 else str(max(1, box))
        if sp["final_third"] is not None and rng.random() < recall:
            add("BALL IN FINAL THIRD", sp["final_third"] - sec(0, 1.5), jitter_end, team, half,
                Side=str(rng.choice(["Left", "Central", "Right"])), **{"Max Players in the box": box_label})
        if sp["in_box"] is not None and rng.random() < recall:
            add("BALL IN THE BOX", sp["in_box"] - sec(0.5, 1.5), sp["in_box"] + sec(2, 4), team, half,
                **{"Direction of ball entry": str(rng.choice(["Horizontal", "Vertical", "Diagonal"])),
                   "Max Players in the box": box_label})
            if rng.random() < recall:
                add("PLAYERS IN THE BOX", sp["in_box"] - sec(0, 1), sp["in_box"] + sec(3, 6), team, half,
                    **{"Max Players in the box": box_label})
        if sp["max_depth"] > 0.8 and rng.random() < 0.35:
            add("CREATING CHANCES", end - sec(4, 7), end + sec(0, 2), team, half)
        if sp["cause"] == "turnover" and rng.random() < 0.5 * recall:
            stop = start + sec(4, 9)
            add("ATTACKING TRANSITION", jitter_start, stop, team, half)
            add("DEFENSIVE TRANSITION", jitter_start, stop, other, half)
        if rng.random() < 0.3:
            d_def = 1.0 - zone
            third = "DEFENSIVE" if d_def < 1 / 3 else ("MIDDLE" if d_def < 2 / 3 else "ATTACKING")
            add("DEFENDING IN %s THIRD" % third, jitter_start, jitter_end, other, half)

    codes = ["BUILD UP", "PROGRESSION", "DEFENDING IN MIDDLE THIRD", "BALL IN FINAL THIRD",
             "ATTACKING TRANSITION", "LONG BALL"]
    for _ in range(int(prm["tag_false"] * len(instances))):          # false positives
        half, first, last = HALF_BLOCKS[int(rng.integers(2))]
        start = int(rng.integers(first, last - 10 * FPS))
        add(str(rng.choice(codes)), start, start + sec(3, 10), int(rng.integers(2)), half)

    instances.sort(key=lambda inst: inst["start"])
    return instances


# ==================================================================================================
# Writers -- the same file formats as the real bundle
# ==================================================================================================

def write_atd_meta(path, home_right_first):
    halves = {h: (a + 1, b + 1) for h, a, b in HALF_BLOCKS}           # ATD frame_id = video + 1
    team_ids = ("home", "away")
    direction = ("left_to_right", "right_to_left")
    home_dir = direction[0] if home_right_first else direction[1]
    away_dir = direction[1] if home_right_first else direction[0]

    def param(name, value):
        return "<ProviderParameter><Name>%s</Name><Value>%s</Value></ProviderParameter>" % (name, value)

    globals_ = "".join([param("first_half_start", halves[1][0]), param("first_half_end", halves[1][1]),
                        param("second_half_start", halves[2][0]), param("second_half_end", halves[2][1]),
                        param("data_type", "atd"), param("tracking_source", "synthetic")])
    teams = "".join(
        '<Team id="%s"><Name>%s</Name><ProviderTeamsParameters>%s</ProviderTeamsParameters></Team>'
        % (team_ids[t], ATD_TEAM_NAMES[t], param("attack_direction_first_half", (home_dir, away_dir)[t]))
        for t in (HOME, AWAY))
    players, channels, refs = [], [], []
    for k in range(N_SLOTS):
        pid = str(k + 1)
        team = team_ids[HOME if k < SLOTS_PER_TEAM else AWAY]
        players.append('<Player id="%s" teamId="%s"><Name>Track %s</Name><ShirtNumber>%s</ShirtNumber>'
                       '<ProviderPlayerParameters>%s%s</ProviderPlayerParameters></Player>'
                       % (pid, team, pid, pid, param("position_type", "Unknown"), param("position_index", -1)))
        for ch, name in (("x", "x"), ("y", "y"), ("s", "speed")):
            channels.append('<PlayerChannel channelId="%s" id="player%s_%s" playerId="%s"/>' % (ch, pid, name, pid))
        refs.append('<SplitRegister separator=",">%s</SplitRegister>' % "".join(
            '<PlayerChannelRef playerChannelId="player%s_%s"/>' % (pid, name) for name in ("x", "y", "speed")))
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata>'
        '<GlobalConfig><FileName>synthetic</FileName><TrackingType>Optical</TrackingType>'
        '<ProviderName>Metrica Sports</ProviderName><FrameRate>%d</FrameRate>'
        '<ProviderGlobalParameters>%s</ProviderGlobalParameters></GlobalConfig>'
        '<Sessions><Session id="session1"><SessionType>Match</SessionType><MatchParameters>'
        '<Score idLocalTeam="home" idVisitingTeam="away"><LocalTeamScore/><VisitingTeamScore/></Score>'
        '</MatchParameters><FieldSize><Width>105</Width><Height>68</Height></FieldSize></Session></Sessions>'
        '<Teams>%s</Teams><Players>%s</Players>'
        '<Devices><Device id="device1"><Name>Device 1</Name><Sensors>'
        '<Sensor id="position"><Name>Position</Name><Channels>'
        '<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>'
        '<Sensor id="speed"><Name>Speed in metres per second</Name><Channels>'
        '<Channel id="s"><Name>speed</Name><Unit>meters per second</Unit></Channel></Channels></Sensor>'
        '</Sensors></Device></Devices><PlayerChannels>%s</PlayerChannels></Metadata>'
        '<DataFormatSpecifications><DataFormatSpecification endFrame="%d" separator=":" startFrame="1">'
        '<StringRegister name="frameCount"/><SplitRegister separator=";">%s</SplitRegister>'
        '<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
        '<BallChannelRef channelId="s"/></SplitRegister></DataFormatSpecification>'
        '</DataFormatSpecifications></main>\n'
        % (FPS, globals_, teams, "".join(players), "".join(channels), N_FRAMES, "".join(refs)))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)


def write_atd_raw(path, slots, slot_speed, ball, ball_speed):
    """One line per video frame: "frame_id:x,y,s;x,y,s;...:x,y,s", coordinates normalised 0-1."""
    nx, ny = slots[..., 0] / L, slots[..., 1] / W
    bx, by = ball[:, 0] / L, ball[:, 1] / W
    empty = "NaN,NaN,NaN"
    buffer = []
    with open(path, "w", newline="\n") as handle:
        for f in range(N_FRAMES):
            groups = [empty] * N_SLOTS
            xs, ys, ss = nx[f].tolist(), ny[f].tolist(), slot_speed[f].tolist()
            for k in np.flatnonzero(~np.isnan(nx[f])).tolist():
                s = ss[k]
                groups[k] = "%.5f,%.5f,%s" % (xs[k], ys[k], "NaN" if s != s else "%.3f" % s)
            if bx[f] == bx[f]:
                s = ball_speed[f]
                ball_group = "%.5f,%.5f,%s" % (bx[f], by[f], "NaN" if s != s else "%.3f" % s)
            else:
                ball_group = empty
            buffer.append("%d:%s:%s\n" % (f + 1, ";".join(groups), ball_group))
            if len(buffer) >= 5000:
                handle.writelines(buffer)
                buffer = []
        handle.writelines(buffer)


def homography_matrix():
    """A fixed broadcast-like camera: normalised pitch (0-1) -> 1920x1080 pixels."""
    src = [(0, 0), (1, 0), (0, 1), (1, 1)]
    dst = [(170, 250), (1750, 250), (-80, 1010), (2000, 1010)]
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.array(rows, dtype=float))[2][-1].reshape(3, 3)
    return h / h[2, 2]


def write_homography(path, sim, rng):
    h = homography_matrix().ravel()
    error = np.where(sim.graded, 0.24 + np.abs(rng.normal(0, 0.06, N_FRAMES)),
                     1.5 + np.abs(rng.normal(0, 0.3, N_FRAMES)))
    table = np.column_stack([np.arange(N_FRAMES), np.tile(h, (N_FRAMES, 1)), error, np.ones(N_FRAMES)])
    header = "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"
    np.savetxt(path, table, delimiter=",", header=header, comments="",
               fmt=["%d"] + ["%.7g"] * 9 + ["%.4f", "%d"])


def write_smart_tagging(path, instances):
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "  <ALL_INSTANCES>"]
    for n, inst in enumerate(instances, 1):
        out.append("    <instance><ID>%d</ID><start>%.2f</start><end>%.2f</end><code>%s</code>"
                   % (n, inst["start"] / FPS, inst["end"] / FPS, inst["code"]))
        for group, text in inst["labels"].items():
            out.append("      <label><group>%s</group><text>%s</text></label>" % (group, text))
        out.append("    </instance>")
    out += ["  </ALL_INSTANCES>", "</file>", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out))


def write_gt(path, gt):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (f, label) for f, label in enumerate(gt.tolist()))


# ==================================================================================================
# Main
# ==================================================================================================

def possession_runs(gt):
    """(count, mean seconds) of maximal H/A runs inside the graded halves."""
    lengths = []
    for _, first, last in HALF_BLOCKS:
        labels = gt[first:last + 1]
        change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        bounds = np.concatenate([[0], change, [len(labels)]])
        lengths += [b - a for a, b in zip(bounds[:-1], bounds[1:]) if labels[a] in ("H", "A")]
    return len(lengths), (float(np.mean(lengths)) / FPS if lengths else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="same seed, same files, byte for byte")
    parser.add_argument("--out", default=None, help="output folder (default: fake_data/seed<SEED>)")
    parser.add_argument("--ball-missing", type=float, default=DEFAULTS["ball_missing"],
                        help="share of graded frames without a ball (default 0.41, as the real data)")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="multiply every tracking error rate: 0 = clean tracking, 2 = twice as bad")
    parser.add_argument("--no-truth", action="store_true", help="skip truth.npz (saves ~40 MB)")
    args = parser.parse_args()

    prm = dict(DEFAULTS)
    prm["ball_missing"] = args.ball_missing
    for key in ("ball_noise_m", "ball_false_rate", "player_noise_m", "player_loss", "swap_same_team",
                "swap_cross_team", "fragment", "duplicate"):
        prm[key] *= args.noise_scale
    out_dir = args.out or os.path.join(ROOT, "fake_data", "seed%d" % args.seed)
    tactical = os.path.join(out_dir, "tactical")
    os.makedirs(tactical, exist_ok=True)
    paths = games.asset_paths(data_dir=out_dir)
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    home_right_first = bool(rng.integers(2))
    first_kick = int(rng.integers(2))
    sim = MatchSim(rng, home_right_first)
    for half, first, last in HALF_BLOCKS:
        sim.run_half(half, first, last, first_kick if half == 1 else 1 - first_kick)
    print("simulated the match            %5.1f s" % (time.time() - t0))

    ball = degrade_ball(sim, rng, prm)
    slots = degrade_players(sim, rng, prm)
    print("degraded the tracking          %5.1f s" % (time.time() - t0))

    write_atd_meta(paths["atd_meta"], home_right_first)
    write_atd_raw(paths["atd_raw"], slots, speeds(slots), ball, speeds(ball))
    write_homography(paths["homography"], sim, rng)
    instances = build_tags(sim, rng, prm)
    write_smart_tagging(paths["smart_tagging"], instances)
    gt_path = os.path.join(out_dir, "%s_possession_gt_fake.csv" % GAME)
    write_gt(gt_path, sim.gt)
    if not args.no_truth:
        code = {"H": 0, "A": 1, "D": 2, "X": -1}
        np.savez_compressed(
            os.path.join(out_dir, "truth.npz"),
            ball=(sim.true_ball / np.array([L, W])).astype(np.float32),        # normalised, clean
            players=(sim.true_pos / np.array([L, W])).astype(np.float32),     # (frame, 22, 2)
            player_team=TEAM.astype(np.int8),
            carrier=sim.carrier_track,                                         # player index or -1
            label=np.array([code[c] for c in sim.gt.tolist()], dtype=np.int8),  # 0 H, 1 A, 2 D, -1 X
            ball_in_air=sim.ball_air)
    print("wrote the bundle               %5.1f s" % (time.time() - t0))

    graded = sim.graded
    n_poss, mean_poss = possession_runs(sim.gt)
    restart_counts = {}
    for r in sim.restarts:
        restart_counts[r["type"]] = restart_counts.get(r["type"], 0) + 1
    stats = {
        "ball_visible_graded": round(float(np.mean(~np.isnan(ball[graded, 0]))), 4),
        "ball_visible_all": round(float(np.mean(~np.isnan(ball[:, 0]))), 4),
        "mean_tracked_objects_graded": round(float(np.mean(np.sum(~np.isnan(slots[graded, :, 0]), axis=1))), 2),
        "possessions": n_poss,
        "mean_possession_s": round(mean_poss, 1),
        "dead_share_graded": round(float(np.mean(sim.gt[graded] == "D")), 4),
        "restarts": restart_counts,
        "goals": [{"team": "HA"[g["team"]], "frame": g["frame"]} for g in sim.goals],
        "shots": len(sim.shots),
        "smart_tagging_instances": len(instances),
    }
    meta = {"seed": args.seed, "generator": "tools/make_fake_data.py", "parameters": prm,
            "home_attacks_right_in_first_half": home_right_first,
            "first_half_kickoff_team": "HA"[first_kick], "stats": stats,
            "note": "Synthetic data. Not real football and not a validation set: see the docstring."}
    with open(os.path.join(out_dir, "fake_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    print("\nfake match written to %s" % os.path.relpath(out_dir, os.getcwd()))
    for key, value in stats.items():
        print("  %-28s %s" % (key, value))
    print("\nuse it:  MTAH_DATA_DIR=%s python starter/baseline.py --out %s"
          % (os.path.relpath(out_dir), os.path.relpath(os.path.join(out_dir, "results"))))


if __name__ == "__main__":
    main()
