"""Popularity ties must not change metric groups across execution backends."""

import pytest
import torch

from warprec.utils.registry import metric_registry


@pytest.mark.parametrize("metric_name", ["ACLT", "APLT", "PopREO", "PopRSP"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_popularity_ties_use_ascending_item_indices(metric_name: str, device: str):
    """A tied block crossing the 80% boundary has the same members everywhere."""
    interactions = torch.ones(1000, device=device)
    interactions[::2] = 2
    metric = metric_registry.get_class(metric_name)(
        k=1, num_users=1, item_interactions=interactions
    )

    # 500 even items contribute 1000 interactions. The strict >80% cutoff
    # includes 201 odd items as well, for 1201 of 1500 interactions.
    expected_tail = torch.arange(403, 1000, 2, device=device)
    torch.testing.assert_close(metric.long_tail, expected_tail, rtol=0, atol=0)
    if hasattr(metric, "short_head"):
        expected_head = torch.cat(
            (
                torch.arange(0, 1000, 2, device=device),
                torch.arange(1, 403, 2, device=device),
            )
        )
        torch.testing.assert_close(metric.short_head, expected_head, rtol=0, atol=0)


def test_popularity_partition_keeps_the_strict_cumulative_cutoff():
    """An exact 80% prefix still includes the next item, as before this fix."""
    metric = metric_registry.get_class("PopREO")(
        k=1, item_interactions=torch.tensor([2.0, 3.0, 5.0])
    )
    torch.testing.assert_close(metric.short_head, torch.tensor([2, 1, 0]))
    torch.testing.assert_close(metric.long_tail, torch.empty(0, dtype=torch.long))
