"""Rank discounts honor metric precision; the float32 path stays unchanged."""

import math
import pytest
import torch
from warprec.utils.registry import metric_registry


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binary_ndcg_requested_precision(dtype):
    """Two hits at ranks 2 and 3 are normalized by ideal ranks 1 and 2."""
    metric = metric_registry.get_class("nDCGRendle2020")(k=3, num_users=1).set_dtype(
        dtype
    )
    metric.update(
        preds=torch.tensor([[1.0, 0.0, 2.0, -torch.inf]]),
        ground=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        user_indices=torch.tensor([0]),
    )
    actual = float(metric.compute()["nDCGRendle2020"].reshape(-1)[0])
    if dtype == torch.float64:
        expected = math.fsum([1 / math.log2(3), 0.5]) / math.fsum(
            [1.0, 1 / math.log2(3)]
        )
        assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
    else:
        discounts = torch.log2(torch.arange(2, 5))
        expected = (torch.tensor([0.0, 1.0, 1.0]) / discounts).sum() / (
            torch.tensor([1.0, 1.0, 0.0]) / discounts
        ).sum()
        assert actual == float(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_discount_normalizer_requested_precision(dtype):
    """The nominal-k novelty normalizer uses the same requested discounts."""
    metric = metric_registry.get_class("nDCGRendle2020")(k=3, num_users=1).set_dtype(
        dtype
    )
    actual = float(metric.discounted_sum(3))
    if dtype == torch.float64:
        assert actual == pytest.approx(
            math.fsum([1.0, 1 / math.log2(3), 0.5]), rel=1e-14, abs=1e-14
        )
    else:
        assert actual == float((1 / torch.log2(torch.arange(2, 5).float())).sum())
