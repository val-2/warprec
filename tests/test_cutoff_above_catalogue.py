"""A cutoff is a maximum even when the catalogue itself is shorter."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from warprec.data.ranking import rank_top_k
from warprec.evaluation import Evaluator
from warprec.utils.registry import metric_registry


def test_cutoff_above_catalogue_preserves_metric_denominators():
    """Three available items at k=8 use Precision's 8 and list means' 3."""
    names = [
        "nDCGRendle2020",
        "nDCG",
        "MAP",
        "Precision",
        "ARP",
        "APLT",
        "NumRetrieved",
        "ItemCoverage",
        "Gini",
        "UserMADRanking",
    ]
    metrics = Evaluator(
        names,
        [8],
        train_set=csr_matrix(np.ones((2, 3))),
        user_cluster=torch.ones(2, dtype=torch.long),
    )
    metrics._compute_metrics_step(
        predictions=torch.tensor([[0.5, 0.4, 0.3], [-torch.inf] * 3]),
        eval_batch=torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        user_indices=torch.tensor([0, 1]),
        candidates=None,
    )
    result = metrics.compute_results()[8]
    for name in ["nDCGRendle2020", "nDCG", "MAP"]:
        torch.testing.assert_close(result[name], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result["Precision"], torch.tensor([1 / 8, 0.0]))
    assert float(result["ARP"][0]) == 2.0
    assert torch.isnan(result["ARP"][1])
    assert float(result["APLT"][0]) == 0.0
    assert torch.isnan(result["APLT"][1])
    torch.testing.assert_close(result["NumRetrieved"], torch.tensor([3.0, 0.0]))
    assert float(result["ItemCoverage"]) == 3.0
    assert float(result["Gini"]) == 0.0


@pytest.mark.parametrize("seeded", [False, True])
def test_direct_ranking_returns_only_catalogue_positions(seeded: bool):
    """No repeated or fabricated index is needed to reach the requested cutoff."""
    generator = torch.Generator().manual_seed(1) if seeded else None
    values, indices = rank_top_k(torch.tensor([[0.5, 0.3, 0.4]]), 8, generator)
    torch.testing.assert_close(values, torch.tensor([[0.5, 0.4, 0.3]]))
    torch.testing.assert_close(indices, torch.tensor([[0, 2, 1]]))


def test_direct_metric_fallback_accepts_a_larger_cutoff():
    """Metric-local top-k derivation agrees with the evaluator's effective width."""
    metric = metric_registry.get_class("nDCGRendle2020")(k=8, num_users=2)
    metric.update(
        torch.tensor([[0.5, 0.4, 0.3]]),
        user_indices=torch.tensor([0]),
        ground=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    assert float(metric.compute()["nDCGRendle2020"][0]) == 1.0
