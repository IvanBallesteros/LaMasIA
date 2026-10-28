"""Per-frame evidence from ATD-like tracking and Smart Tagging. One row per frame, NaN where a
signal has no evidence (the classifier handles missing values itself).

The inputs are the same for the challenge match (tools/heuristic_pipeline.load_tracking) and for
the datasets (tools/build_datasets.load_match): positions in metres from the top-left corner, x
along the pitch, frames on the match's own timeline, slot_team 0 = home. Nothing here reads a
label.

Feature groups, any of which can be left out:

    ball     where the ball is and how fast it moves
    near     who is near the ball right now
    context  who has been near the ball over the last / next seconds (centred windows)
    players  how fast the players move, where each team is and whether it is advancing
    gap      how long the ball has been / will be missing -- DEPENDS ON WHERE THE TRACKER LOSES
             THE BALL, which differs between IDSSE and Metrica and is not known yet for the real
             match (tools/ball_gap_diagnostic.py). Keep a model without it until that is settled.
    tags     Smart Tagging SET PIECES windows
"""

import numpy as np
import pandas as pd

FPS = 25
GROUPS = ("ball", "near", "context", "players", "gap", "tags")
NEAR_M = 3.0                 # "near the ball" for player counts
STILL_MS = 1.5               # a player slower than this is standing
MAX_PLAYER_MS = 12.0         # faster is an identity switch, not running
MAX_BALL_MS = 40.0           # the hardest shots stay below this
CONTEXT_S = (0.5, 2.0, 5.0)  # half-widths of the centred windows
TAG_HORIZON_S = 30.0
COLUMNS = {
    "ball": ["ball_speed", "ball_x_home_attack", "ball_to_touchline", "ball_to_goalline", "ball_outside",
             "ball_speed_1s"],
    "near": ["d_home", "d_away", "d_diff", "n_home_near", "n_away_near", "nearest_home"],
    "context": [c % s for s in CONTEXT_S for c in ("nearest_home_%gs", "ball_seen_%gs", "d_diff_%gs")],
    "players": ["speed_all", "speed_home", "speed_away", "still_share", "n_objects", "territory_home",
                "territory_away", "advance_home", "advance_away", "speed_all_2s"],
    "gap": ["ball_seen", "gap_since_s", "gap_until_s", "gap_length_s"],
    "tags": ["in_set_piece", "set_piece_team", "to_set_piece_s"],
}


def _speed(xy, k=2):
    """Speed in m/s over +/-k frames, NaN where unknown. xy: (n, 2) or (n, slots, 2)."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) / (2 * k) * FPS
    return out


def _window_mean(values, half):
    """Centred mean over +/-half frames ignoring NaN, and the share of the window with a value."""
    ok = ~np.isnan(values)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0), dtype=np.float64)))
    k = np.concatenate(([0.0], np.cumsum(ok, dtype=np.float64)))
    i = np.arange(len(values))
    lo, hi = np.clip(i - half, 0, len(values)), np.clip(i + half + 1, 0, len(values))
    total, count = s[hi] - s[lo], k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return mean.astype(np.float32), (count / (hi - lo)).astype(np.float32)


def _attack_sign(players, slot_team, halves, L):
    """+1 where the home team attacks towards x = L, -1 towards x = 0 (per half, from where the
    home players stand on average: a team defends its own half)."""
    sign = np.zeros(len(players), dtype=np.float32)
    home = slot_team == 0
    for a, b in halves:
        with np.errstate(invalid="ignore"):
            mean_x = np.nanmean(players[a:b + 1][:, home, 0])
        sign[a:b + 1] = 1.0 if mean_x < L / 2 else -1.0
    return sign


def _gap_lengths(missing):
    """Seconds since the ball was last seen, until it is seen again, and the current gap's length."""
    n = len(missing)
    idx = np.arange(n)
    last = np.where(~missing, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(~missing, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, idx - last, np.nan) / FPS
    until = np.where(nxt < n, nxt - idx, np.nan) / FPS
    return since.astype(np.float32), until.astype(np.float32), (since + until).astype(np.float32)


def frame_features(players, slot_team, ball, pitch, halves, set_pieces=(), groups=GROUPS):
    """DataFrame (one row per frame) of the requested feature groups.

    players [n, slots, 2], ball [n, 2] in metres (NaN = missing); slot_team [slots] 0/1;
    pitch (L, W); halves [(first, last)] frames in play; set_pieces [(lo, hi, team)] with team
    'H', 'A' or None (unknown, or N/A as on every throw-in).
    """
    L, W = pitch
    n = len(ball)
    slot_team = np.asarray(slot_team)
    home, away = slot_team == 0, slot_team == 1
    attack = _attack_sign(players, slot_team, halves, L)
    seen = ~np.isnan(ball[:, 0])
    cols = {}

    if "ball" in groups:
        x, y = ball[:, 0], ball[:, 1]
        cols["ball_speed"] = np.minimum(_speed(ball), MAX_BALL_MS)       # jumps are false detections
        cols["ball_x_home_attack"] = (x - L / 2) * attack / (L / 2)
        cols["ball_to_touchline"] = np.minimum(y, W - y)
        cols["ball_to_goalline"] = np.minimum(x, L - x)
        cols["ball_outside"] = np.maximum.reduce([-x, x - L, -y, y - W, np.zeros(n, dtype=np.float32)])
        cols["ball_speed_1s"], _ = _window_mean(cols["ball_speed"], FPS)

    need_near = "near" in groups or "context" in groups
    if need_near:
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(players - ball[:, None, :], axis=2)
        d_home = np.where(np.isnan(d[:, home]), np.inf, d[:, home]).min(axis=1)
        d_away = np.where(np.isnan(d[:, away]), np.inf, d[:, away]).min(axis=1)
        d_home = np.where(seen & np.isfinite(d_home), d_home, np.nan).astype(np.float32)
        d_away = np.where(seen & np.isfinite(d_away), d_away, np.nan).astype(np.float32)
        with np.errstate(invalid="ignore"):
            nearest_home = np.where(np.isnan(d_home) | np.isnan(d_away), np.nan, (d_home < d_away).astype(np.float32))
    if "near" in groups:
        cols["d_home"], cols["d_away"] = d_home, d_away
        cols["d_diff"] = d_home - d_away
        with np.errstate(invalid="ignore"):
            cols["n_home_near"] = np.where(seen, np.sum(d[:, home] < NEAR_M, axis=1), np.nan).astype(np.float32)
            cols["n_away_near"] = np.where(seen, np.sum(d[:, away] < NEAR_M, axis=1), np.nan).astype(np.float32)
        cols["nearest_home"] = nearest_home
    if "context" in groups:
        for s in CONTEXT_S:
            mean, cover = _window_mean(nearest_home, int(s * FPS))
            cols["nearest_home_%gs" % s], cols["ball_seen_%gs" % s] = mean, cover
            cols["d_diff_%gs" % s], _ = _window_mean(d_home - d_away, int(s * FPS))

    if "players" in groups:
        v = _speed(players)
        v = np.where(v > MAX_PLAYER_MS, np.nan, v)
        with np.errstate(invalid="ignore"), _quiet():
            cols["speed_all"] = np.nanmedian(v, axis=1)
            cols["speed_home"] = np.nanmean(v[:, home], axis=1)
            cols["speed_away"] = np.nanmean(v[:, away], axis=1)
            cols["still_share"] = np.nanmean(np.where(np.isnan(v), np.nan, v < STILL_MS), axis=1)
            cx_home = np.nanmean(players[:, home, 0], axis=1)
            cx_away = np.nanmean(players[:, away, 0], axis=1)
        cols["n_objects"] = np.sum(~np.isnan(players[:, :, 0]), axis=1).astype(np.float32)
        cols["territory_home"] = (cx_home - L / 2) * attack / (L / 2)
        cols["territory_away"] = -(cx_away - L / 2) * attack / (L / 2)
        adv_home = np.full(n, np.nan, dtype=np.float32)
        adv_away = np.full(n, np.nan, dtype=np.float32)
        adv_home[FPS:-FPS] = (cx_home[2 * FPS:] - cx_home[:-2 * FPS]) / 2.0 * attack[FPS:-FPS]
        adv_away[FPS:-FPS] = -(cx_away[2 * FPS:] - cx_away[:-2 * FPS]) / 2.0 * attack[FPS:-FPS]
        cols["advance_home"], cols["advance_away"] = adv_home, adv_away
        cols["speed_all_2s"], _ = _window_mean(cols["speed_all"].astype(np.float32), 2 * FPS)

    if "gap" in groups:
        since, until, length = _gap_lengths(~seen)
        cols["ball_seen"] = seen.astype(np.float32)
        cols["gap_since_s"], cols["gap_until_s"], cols["gap_length_s"] = since, until, length

    if "tags" in groups:
        inside = np.zeros(n, dtype=np.float32)
        team = np.zeros(n, dtype=np.float32)
        to_start = np.full(n, TAG_HORIZON_S, dtype=np.float32)
        for lo, hi, t in sorted(set_pieces, key=lambda w: w[0]):
            lo, hi = max(int(lo), 0), min(int(hi), n)
            if lo >= n:
                continue
            inside[lo:hi] = 1.0
            team[lo:hi] = {"H": 1.0, "A": -1.0}.get(t, 0.0)
            a, b = max(lo - int(TAG_HORIZON_S * FPS), 0), min(lo + int(TAG_HORIZON_S * FPS), n)
            dist = (np.arange(a, b) - lo) / FPS
            to_start[a:b] = np.where(np.abs(dist) < np.abs(to_start[a:b]), dist, to_start[a:b])
        cols["in_set_piece"], cols["set_piece_team"], cols["to_set_piece_s"] = inside, team, to_start

    assert sorted(cols) == sorted(c for g in groups for c in COLUMNS[g])
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)
