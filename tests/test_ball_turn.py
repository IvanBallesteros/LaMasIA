#!/usr/bin/env python3
"""The ball turn features on a hand-made ball: python tests/test_ball_turn.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mtah.features import frame_features  # noqa: E402

L, W = 100.0, 60.0


def main():
    n = 300
    team = np.array([0, 1])
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0], players[:, 1] = (20, 30), (80, 30)                  # home defends x = 0
    t = np.arange(n, dtype=np.float32)
    ball = np.stack([np.where(t < 150, 20 + 0.4 * t, 80 - 0.4 * (t - 150)), np.full(n, 30.0)], axis=1).astype(np.float32)
    ball[250:, :] = np.nan                                              # missing ball gives NaN

    f = frame_features(players, team, ball, (L, W), [(0, n - 1)], groups=("turn",))
    assert np.isclose(f["ball_dir_past_0.5s"].iloc[100], 10), "towards the away goal at 10 m/s"
    assert np.isclose(f["ball_dir_next_0.5s"].iloc[200], -10), "back towards the home goal"
    assert np.isclose(f["ball_dir_jump_0.5s"].iloc[150], -20), "turned around: points to the away team"
    assert abs(f["ball_dir_jump_0.5s"].iloc[100]) < 1e-3, "no turn while it runs straight"
    assert np.isnan(f["ball_dir_next_0.5s"].iloc[245]) and np.isnan(f["ball_dir_past_2s"].iloc[5])

    flipped = frame_features(players[:, ::-1], team, ball, (L, W), [(0, n - 1)], groups=("turn",))
    assert np.isclose(flipped["ball_dir_past_0.5s"].iloc[100], -10), "home attacks the other way"
    print("ok")


if __name__ == "__main__":
    main()
