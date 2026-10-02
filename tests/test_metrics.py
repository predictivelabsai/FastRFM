"""Metric definitions."""

import numpy as np

from fastrfm.metrics import auroc, hit_rate, mae


def test_auroc_extremes_and_ties():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 1], [0.4, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5


def test_auroc_undefined_without_both_classes():
    assert np.isnan(auroc([1, 1, 1], [0.2, 0.4, 0.9]))


def test_mae_and_hit_rate():
    assert mae([0.0, 10.0], [0.0, 6.0]) == 2.0
    assert hit_rate([1, 2, 3], [1, 0, 3]) == 2 / 3
