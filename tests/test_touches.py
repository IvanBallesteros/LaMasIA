#!/usr/bin/env python3
"""The touch features on a ball bouncing between two players: python tests/test_touches.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mtah.features import frame_features  # noqa: E402


def main():
    n = 500
    frames = np.arange(n)
    x = 40 + 0.4 * np.abs((frames + 100) % 200 - 100)    # 10 m/s, turning at x = 80 and x = 40
    ball = np.stack([x, np.full(n, 30.0)], axis=1).astype(np.float32)   # turns at 100 (home), 200 (away), ...
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0], players[:, 1] = (80.0, 30.0), (40.0, 30.0)
    f = frame_features(players, np.array([0, 1]), ball, (100.0, 60.0), [(0, n - 1)], groups=("touches",))

    mid = f.iloc[150]
    assert mid["tch_last_home"] == 1 and mid["tch_last_s"] == 2 and mid["tch_last_run"] == 1
    assert mid["tch_next_home"] == 0 and mid["tch_next_s"] == 2 and mid["tch_next_run"] == 1
    assert np.isnan(f["tch_last_home"].iloc[50]), "no touch yet"
    assert np.isnan(f["tch_next_home"].iloc[450]), "no touch left"
    at = f.iloc[205]                                     # 2 s window: the touch at 200 only, then the one at 300 is too far
    assert at["tch_away_past"] == 1 and at["tch_home_past"] == 0
    assert at["tch_home_next"] == 0 and at["tch_away_next"] == 0

    straight = ball.copy()
    straight[:, 0] = 0.2 * frames                        # a ball rolling past nobody touches nothing
    g = frame_features(players, np.array([0, 1]), straight, (100.0, 60.0), [(0, n - 1)], groups=("touches",))
    assert g["tch_last_s"].isna().all()
    print("ok")


if __name__ == "__main__":
    main()
