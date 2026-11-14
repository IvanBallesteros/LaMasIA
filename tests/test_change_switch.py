#!/usr/bin/env python3
"""The input-dependent switch cost on a hand-made sequence: python tests/test_change_switch.py

A home possession with two short, weakly seen away spells inside it. With a fixed switch cost the
decoder ignores both. With the change detector high at the edges of the first spell only, change_k
lets that one through and still ignores the other. change_k 0 must be the plain decoder exactly.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from mtah.change_detector import direct_changes, near_change  # noqa: E402
from mtah.decoder import change_shift, decode  # noqa: E402


def main():
    n, halves, costs = 1000, [(0, 999)], [[32, 64, 32, 0]]
    proba = np.tile([0.90, 0.05, 0.05], (n, 1))
    seen, unseen = slice(400, 440), slice(700, 740)
    proba[seen] = proba[unseen] = (0.35, 0.60, 0.05)            # 40 frames of weak evidence for away

    base = 0.03
    p_change = np.full(n, 0.005)
    for edge in (400, 440):
        p_change[edge - 10:edge + 10] = 0.6                      # the detector fires at the first spell only
    shift = change_shift(p_change, base)
    assert abs(change_shift(np.array([base]), base)[0]) < 1e-12, "the base rate is the zero"

    plain = decode(proba, halves, [], costs, 0)[0]
    assert (plain == "H").all(), "a fixed switch cost ignores both spells"
    assert np.array_equal(decode(proba, halves, [], costs, 0, 5, shift, [0])[0], plain), "change_k 0 is the plain decoder"

    k8 = decode(proba, halves, [], costs, 0, 5, shift, [8])[0]
    assert (k8[405:435] == "A").all(), "the spell the detector points at comes through"
    assert (k8[unseen] == "H").all(), "the other one stays out"
    assert (k8[:390] == "H").all() and (k8[450:] == "H").all()

    labels = np.array(list("HHHAAADDAAX"))
    assert list(direct_changes(labels)) == [3], "only H <-> A with no stoppage in between"
    assert list(np.flatnonzero(near_change(labels, 1))) == [2, 3, 4]
    print("ok")


if __name__ == "__main__":
    main()
