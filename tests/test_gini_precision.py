"""Gini must preserve configured precision after accumulating item counts."""

from fractions import Fraction

import pytest
import torch

from warprec.evaluation.metrics.diversity.gini_index import Gini


def test_gini_honors_float64_metric_precision():
    """Compare a nonuniform exposure distribution with exact rational arithmetic."""
    counts = [(i * 37) % 23 + 1 for i in range(1, 1200)] + [0] * 411
    ordered = sorted(counts)
    numerator = sum((2 * (j + 1) - len(counts) - 1) * n for j, n in enumerate(ordered))
    expected = float(Fraction(numerator, sum(counts) * (len(counts) - 1)))
    metric = Gini(k=10, num_items=len(counts)).set_dtype(torch.float64)
    metric.item_counts.copy_(torch.tensor(counts, dtype=torch.float64))
    metric.free_norm.fill_(sum(counts))

    assert metric.compute()["Gini"] == pytest.approx(expected, rel=1e-14, abs=1e-14)
