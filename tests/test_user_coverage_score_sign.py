"""Recommendation coverage depends on availability, not score sign."""

import torch

from warprec.utils.registry import metric_registry


def test_user_coverage_at_n_counts_finite_negative_and_zero_scores():
    """Two valid scores still form a two-item list when neither is positive."""
    metric = metric_registry.get_class("UserCoverageAtN")(k=2)
    metric.update(
        torch.tensor([[-2.0, 0.0, -torch.inf], [-torch.inf, -1.0, -torch.inf]])
    )

    assert metric.compute()["UserCoverageAtN"] == 1.0
