"""APLT supports explicitly requested precision without changing its default."""

import pytest
import torch
from warprec.utils.registry import metric_registry


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aplt_requested_precision(dtype):
    """One long-tail recommendation among three has exact share 1/3."""
    metric = metric_registry.get_class("APLT")(
        k=3, num_users=1, item_interactions=torch.tensor([8.0, 4.0, 2.0, 1.0, 1.0, 0.0])
    ).set_dtype(dtype)
    metric.update(
        preds=torch.zeros(1, 6),
        user_indices=torch.tensor([0]),
        ground=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        top_3_indices=torch.tensor([[0, 1, 4]]),
    )
    actual = float(metric.compute()["APLT"].reshape(-1)[0])
    if dtype == torch.float64:
        assert actual == pytest.approx(1 / 3, rel=1e-14, abs=1e-14)
    else:
        assert actual == float(torch.tensor(1.0, dtype=torch.float32) / 3)
