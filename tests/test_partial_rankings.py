"""Unavailable top-k slots earn no relevance, exposure or recommendation count.

The two available items are 2 (head, popularity 2) and 4 (tail, popularity 1).
Every other item is masked. The second evaluated user has an empty list.
"""

import math

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from warprec.evaluation import Evaluator
from warprec.utils.registry import metric_registry


def evaluator(names: list[str]) -> Evaluator:
    """Build the popularity and cluster lookups without training a model."""
    counts = [8, 4, 2, 1, 1, 0]
    rows = np.concatenate([np.arange(2, 2 + count) for count in counts])
    columns = np.repeat(np.arange(6), counts)
    train = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(10, 6))
    return Evaluator(
        names,
        [3],
        train_set=train,
        item_cluster=torch.tensor([1, 1, 1, 2, 2, 2, 0]),
        user_cluster=torch.ones(10, dtype=torch.long),
        feature_lookup=torch.cat((torch.eye(6), torch.zeros(1, 6))),
    )


def update_partial(metrics: Evaluator, sampled: bool = False) -> dict:
    """Exercise the actual evaluator step on a partial and an empty list."""
    if sampled:
        # Index 6 is sampled-evaluation padding, outside the six-item catalogue.
        scores = torch.tensor([[0.4, 0.3, -torch.inf], [-torch.inf] * 3])
        target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        candidates = torch.tensor([[2, 4, 6], [0, 1, 6]])
    else:
        scores = torch.tensor(
            [
                [-torch.inf, -torch.inf, 0.4, -torch.inf, 0.3, -torch.inf],
                [-torch.inf] * 6,
            ]
        )
        # Every masked item is relevant, so an invalid slot always creates a
        # false hit unless relevance is gated by recommendation validity.
        target = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0], [1.0] * 6])
        candidates = None
    metrics._compute_metrics_step(
        predictions=scores,
        eval_batch=target,
        user_indices=torch.tensor([0, 1]),
        candidates=candidates,
    )
    return metrics.compute_results()[3]


D2 = 1 / math.log2(3)
DISCOUNT_SUM = 1 + D2 + 0.5


@pytest.mark.parametrize(
    "name,first,second",
    [
        ("nDCGRendle2020", D2 / DISCOUNT_SUM, 0.0),
        ("nDCG", D2 / DISCOUNT_SUM, 0.0),
        ("Precision", 1 / 3, 0.0),
        ("Recall", 1 / 5, 0.0),
        ("HitRate", 1.0, 0.0),
        ("APLT", 0.5, float("nan")),
        ("ARP", 1.5, float("nan")),
        ("ACLT", 1.0, 0.0),
        ("NumRetrieved", 2.0, 0.0),
        ("SRecall", 1 / 5, 0.0),
        ("EPC", 0.9 * D2 / DISCOUNT_SUM, 0.0),
        ("EFD", 4 * D2 / DISCOUNT_SUM, 0.0),
    ],
)
def test_partial_lists_use_explicit_per_user_denominators(
    name: str, first: float, second: float
):
    """Accuracy retains nominal k; list means use actual m and exclude empty lists."""
    result = update_partial(evaluator([name]))[name]
    assert float(result[0]) == pytest.approx(first, abs=1e-6)
    if math.isnan(second):
        assert torch.isnan(result[1])
    else:
        assert float(result[1]) == pytest.approx(second, abs=1e-6)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ItemCoverage", 2.0),
        ("UserCoverage", 1.0),
        ("Gini", 0.8),
        ("ShannonEntropy", math.log(2)),
        ("PopREO", 1.0),
        ("PopRSP", 0.0),
        ("RSP", 0.0),
        ("REO", 1.0),
        ("ItemMADRanking", 3.0),
        ("ItemMADRating", 0.0),
        ("UserMADRating", 0.0),
        ("UserMADRanking", 0.0),
    ],
)
def test_unavailable_slots_do_not_enter_global_exposure(name: str, expected: float):
    """Only two actual recommendations enter catalogue and group statistics."""
    result = update_partial(evaluator([name]))
    assert float(result[name]) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "name", ["nDCGRendle2020", "APLT", "ARP", "ItemCoverage", "Gini", "PopREO"]
)
def test_sampled_padding_never_becomes_a_catalogue_recommendation(name: str):
    """An unavailable padding ID must neither index outside lookups nor count."""
    result = update_partial(evaluator([name]), sampled=True)[name]
    if name == "nDCGRendle2020":
        assert float(result[0]) == pytest.approx(D2, abs=1e-6)
        assert float(result[1]) == 0.0
    elif name in ("APLT", "ARP"):
        assert float(result[0]) == pytest.approx(0.5 if name == "APLT" else 1.5)
        assert torch.isnan(result[1])
    else:
        assert float(result) == pytest.approx(
            {"ItemCoverage": 2.0, "Gini": 0.8, "PopREO": 1.0}[name]
        )


def test_empty_catalogue_exposure_has_no_inequality_distribution():
    """Empty lists mean zero coverage, undefined ARP/APLT/Gini, and zero accuracy."""
    metrics = evaluator(["ItemCoverage", "Gini", "APLT", "ARP", "nDCGRendle2020"])
    metrics._compute_metrics_step(
        predictions=torch.full((2, 6), -torch.inf),
        eval_batch=torch.ones(2, 6),
        user_indices=torch.tensor([0, 1]),
        candidates=None,
    )
    result = metrics.compute_results()[3]
    assert float(result["ItemCoverage"]) == 0
    assert math.isnan(float(result["Gini"]))
    assert torch.isnan(result["APLT"][:2]).all()
    assert torch.isnan(result["ARP"][:2]).all()
    torch.testing.assert_close(result["nDCGRendle2020"][:2], torch.zeros(2))


def test_metric_fallback_uses_the_same_validity_rule_as_the_evaluator():
    """Direct metric updates also exclude masked relevant items."""
    metric = metric_registry.get_class("nDCGRendle2020")(k=3, num_users=2)
    metric.update(
        torch.tensor([[-torch.inf, -torch.inf, 0.4, -torch.inf, 0.3, -torch.inf]]),
        user_indices=torch.tensor([0]),
        ground=torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
    )
    assert float(metric.compute()["nDCGRendle2020"][0]) == pytest.approx(
        D2 / DISCOUNT_SUM
    )


def test_empty_context_does_not_poison_a_nonempty_context_for_the_same_user():
    """Undefined list averages are excluded at the same granularity as accumulation."""
    metrics = evaluator(["APLT", "ARP"])
    scores = torch.tensor(
        [[-torch.inf] * 6, [-torch.inf, -torch.inf, 0.4, -torch.inf, 0.3, -torch.inf]]
    )
    metrics._compute_metrics_step(
        predictions=scores,
        eval_batch=torch.ones(2, 6),
        user_indices=torch.tensor([0, 0]),
        candidates=None,
    )
    result = metrics.compute_results()[3]
    assert float(result["APLT"][0]) == 0.5
    assert float(result["ARP"][0]) == 1.5
