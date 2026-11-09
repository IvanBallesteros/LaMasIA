#!/usr/bin/env python3
"""The team shape features on hand-made teams: python tests/test_team_shape.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mtah.features import frame_features  # noqa: E402

L, W = 100.0, 60.0


def main():
    n = 400
    team = np.array([0, 0, 0, 1, 1, 1])
    players = np.zeros((n, 6, 2), dtype=np.float32)
    players[:, 0], players[:, 1], players[:, 2] = (5, 30), (30, 20), (40, 40)      # home defends x = 0
    players[:, 3], players[:, 4], players[:, 5] = (95, 30), (60, 10), (70, 50)
    t = np.arange(n, dtype=np.float32)
    move = np.where(t < 200, t, 400 - t) * 0.2                                     # 5 m/s, turning at frame 200
    players[:, 1:3, 0] += move[:, None]
    ball = np.full((n, 2), np.nan, dtype=np.float32)                               # shape needs no ball

    f = frame_features(players, team, ball, (L, W), [(0, n - 1)], groups=("shape",))
    r = f.iloc[100]
    assert np.isclose(r["shape_width_home"], np.std([30, 20, 40]))
    assert np.isclose(r["shape_line_home"], 30 + 20 - 50), "second deepest home player, in its attack direction"
    assert np.isclose(r["shape_line_away"], 50 - 70), "the away team attacks the other way"
    assert np.isclose(r["shape_forward_home"], 5 * 2 / 3, atol=0.01) and r["shape_forward_away"] == 0
    assert f["shape_turn_home"].iloc[200] < -5 and abs(f["shape_turn_home"].iloc[100]) < 0.01, "the turn at frame 200"
    assert r["shape_gap"] < 0, "home stands in its own half"
    print("ok")


if __name__ == "__main__":
    main()
