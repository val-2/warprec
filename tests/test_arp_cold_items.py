"""ARP reports observed training counts, including zero for cold items."""

import pytest
import torch
from warprec.utils.registry import metric_registry


def test_arp_preserves_zero_training_popularity():
    """A cold item's interaction count is zero, not a smoothed count of one."""
    metric = metric_registry.get_class("ARP")(
        k=2, num_users=1, item_interactions=torch.tensor([5.0, 0.0, 1.0])
    )
    metric.update(
        preds=torch.zeros(1, 3),
        user_indices=torch.tensor([0]),
        ground=torch.tensor([[1.0, 0.0, 0.0]]),
        top_2_indices=torch.tensor([[0, 1]]),
    )

    assert float(metric.compute()["ARP"].reshape(-1)[0]) == pytest.approx(2.5)
