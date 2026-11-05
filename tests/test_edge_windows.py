#!/usr/bin/env python3
"""The before / after windows on a hand-made change of possession: python tests/test_edge_windows.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mtah.features import frame_features  # noqa: E402


def main():
    n, turnover = 400, 200
    ball = np.full((n, 2), (50.0, 30.0), dtype=np.float32)
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0], players[:, 1] = (50.0, 31.0), (50.0, 35.0)    # home next to the ball...
    players[turnover:, 0], players[turnover:, 1] = (50.0, 35.0), (50.0, 31.0)    # ...then away
    team = np.array([0, 1])
    f = frame_features(players, team, ball, (100.0, 60.0), [(0, n - 1)], groups=("near", "context", "edges"))

    at = f.iloc[turnover]
    assert at["home_past_2s"] == 1 and at["home_next_2s"] == 0 and at["home_jump_2s"] == -1
    assert 0.4 < at["nearest_home_2s"] < 0.6, "the centred window blurs the same frame"
    assert at["d_diff_jump_2s"] == 8
    assert f["home_jump_2s"].iloc[100] == 0 and f["home_jump_2s"].iloc[300] == 0, "nothing away from the change"
    assert np.isnan(f["home_past_2s"].iloc[0]) and np.isnan(f["home_next_2s"].iloc[-1]), "no frames on that side"

    ball[turnover - 5:turnover + 5] = np.nan                     # a missing ball is skipped, not counted
    g = frame_features(players, team, ball, (100.0, 60.0), [(0, n - 1)], groups=("edges",))
    assert g["home_past_0.5s"].iloc[turnover] == 1 and g["home_next_0.5s"].iloc[turnover] == 0
    print("ok")


if __name__ == "__main__":
    main()
