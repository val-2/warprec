"""Undefined opportunity rates must not be reported as perfect parity."""

import math
import pytest
import torch
from warprec.utils.registry import metric_registry


@pytest.mark.parametrize(
    "relevant,recommended",
    [([1.0, 0.0, 0.0], 0), ([0.0, 0.0, 1.0], 2), ([1.0, 0.0, 1.0], 1)],
)
def test_popreo_is_undefined_with_a_missing_group_or_no_hits(
    relevant: list[float], recommended: int
):
    """A missing opportunity rate or two zero hit rates cannot mean parity."""
    metric = metric_registry.get_class("PopREO")(
        k=1, item_interactions=torch.tensor([7.0, 2.0, 1.0])
    )
    target = torch.tensor([relevant])
    indices = torch.tensor([[recommended]])
    metric.update(
        preds=torch.zeros(1, 3),
        binary_relevance=target,
        top_1_indices=indices,
        top_1_binary_relevance=target.gather(1, indices),
    )

    result = metric.compute()
    assert set(result) == {"PopREO"}
    assert math.isnan(result["PopREO"])
