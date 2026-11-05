#!/usr/bin/env python3
"""The memory features on a hand-made sequence: python tests/test_gap_memory.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mtah.features import COLUMNS, frame_features  # noqa: E402

L, W = 100.0, 60.0


def sequence():
    """One half of 300 frames. The ball is missing in 100-199: a home player is next to it before
    the gap, an away player after it. Home defends the left side."""
    n = 300
    ball = np.tile(np.array([[60.0, 30.0]], dtype=np.float32), (n, 1))
    ball[100:200] = np.nan
    ball[:100, 1] = 5.0                                  # near the touchline before the gap
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = (30.0, 30.0)                         # home
    players[:, 1] = (70.0, 30.0)                         # away
    players[:100, 0] = ball[:100] + (1.0, 0.0)
    players[200:, 1] = ball[200:] + (1.0, 0.0)
    return players, np.array([0, 1]), ball


def main():
    players, team, ball = sequence()
    f = frame_features(players, team, ball, (L, W), [(0, 299)], groups=("memory",))
    assert list(f.columns) == sorted(f.columns, key=COLUMNS["memory"].index)
    assert f.iloc[:100].isna().all().all() and f.iloc[200:].isna().all().all(), "NaN where the ball is seen"
    assert (f["mem_last_home"].iloc[100:200] == 1).all() and (f["mem_next_home"].iloc[100:200] == 0).all()
    assert (f["mem_same"].iloc[100:200] == 0).all()
    blend = f["mem_blend"].iloc[100:200].to_numpy()
    assert blend[0] > 0.98 and blend[-1] < 0.02 and np.all(np.diff(blend) < 0), "from the last side to the next"
    assert np.isclose(f["mem_last_to_touchline"].iloc[150], 5) and np.isclose(f["mem_next_to_touchline"].iloc[150], 30)
    assert np.isclose(f["mem_last_x_home_attack"].iloc[150], 0.2)

    # A gap cut by the end of a half only knows its first side, and nothing crosses the half-time.
    f = frame_features(players, team, ball, (L, W), [(0, 149), (150, 299)], groups=("memory",))
    assert (f["mem_last_home"].iloc[100:150] == 1).all() and f["mem_next_home"].iloc[100:150].isna().all()
    assert f["mem_last_home"].iloc[150:200].isna().all() and (f["mem_next_home"].iloc[150:200] == 0).all()
    assert f["mem_blend"].iloc[100:200].isna().all()
    print("ok")


if __name__ == "__main__":
    main()
