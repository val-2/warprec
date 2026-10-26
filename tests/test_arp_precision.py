"""ARP supports explicitly requested precision without changing its default."""

import pytest
import torch
from warprec.utils.registry import metric_registry


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arp_requested_precision(dtype):
    """The exact mean of popularity counts 1, 2 and 4 is 7/3."""
    metric = metric_registry.get_class("ARP")(
        k=3, num_users=1, item_interactions=torch.tensor([1.0, 2.0, 4.0, 0.0])
    ).set_dtype(dtype)
    metric.update(
        preds=torch.zeros(1, 4),
        user_indices=torch.tensor([0]),
        ground=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        top_3_indices=torch.tensor([[0, 1, 2]]),
    )
    actual = float(metric.compute()["ARP"].reshape(-1)[0])
    if dtype == torch.float64:
        assert actual == pytest.approx(7 / 3, rel=1e-14, abs=1e-14)
    else:
        assert actual == float(torch.tensor(7.0, dtype=torch.float32) / 3)
