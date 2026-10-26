"""A file of recommendations must not invent zero-scored missing pairs."""

from pathlib import Path

import pytest
import torch
from scipy.sparse import csr_matrix

from warprec.evaluation import Evaluator
from warprec.recommenders.proxy import ProxyRecommender


def proxy(
    tmp_path: Path, header: bool = True, include_last: bool = False
) -> ProxyRecommender:
    """Replay one negative and one explicitly zero-scored recommendation."""
    file = tmp_path / "recommendations.tsv"
    file.write_text(
        ("user_id\titem_id\trating\n" if header else "")
        + "10\t300\t-2\n10\t100\t0\n"
        + ("10\t500\t-3\n" if include_last else "")
    )
    info = {
        "n_users": 2,
        "n_items": 5,
        "user_mapping": {10: 0, 20: 1},
        "item_mapping": {100: 0, 200: 1, 300: 2, 400: 3, 500: 4},
    }
    return ProxyRecommender(
        {"recommendation_file": str(file), "separator": "\t", "header": header}, info
    )


@pytest.mark.parametrize("header", [True, False])
def test_missing_pairs_are_unavailable_but_explicit_zero_is_retained(
    tmp_path: Path, header: bool
):
    """Unlisted items cannot outrank the negative item; an absent user is empty."""
    model = proxy(tmp_path, header)
    scores = model.predict(torch.tensor([0, 1]))
    expected = torch.tensor(
        [[0.0, -torch.inf, -2.0, -torch.inf, -torch.inf], [-torch.inf] * 5]
    )
    torch.testing.assert_close(scores, expected)
    # Repeated reads do not change sparse presence or score values.
    torch.testing.assert_close(model.predict(torch.tensor([0, 1])), expected)


def test_sampled_proxy_padding_and_unlisted_candidates_stay_unavailable(tmp_path: Path):
    """Sampling cannot turn a padding index into the final real item's score."""
    model = proxy(tmp_path, include_last=True)
    sampled = model.predict(
        torch.tensor([0, 1]), item_indices=torch.tensor([[0, 2, 5], [2, 4, 5]])
    )
    torch.testing.assert_close(
        sampled, torch.tensor([[0.0, -2.0, -torch.inf], [-torch.inf] * 3])
    )


def test_proxy_lists_reach_evaluator_without_fabricated_hits_or_exposure(
    tmp_path: Path,
):
    """A missing user's relevant target is a miss, not a zero-score suggestion."""
    model = proxy(tmp_path)
    metrics = Evaluator(
        ["nDCGRendle2020", "Recall", "ItemCoverage", "NumRetrieved", "UserCoverage"],
        [3],
        train_set=csr_matrix((2, 5)),
    )
    metrics._compute_metrics_step(
        predictions=model.predict(torch.tensor([0, 1])),
        eval_batch=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]),
        user_indices=torch.tensor([0, 1]),
        candidates=None,
    )
    result = metrics.compute_results()[3]
    torch.testing.assert_close(result["Recall"], torch.tensor([1.0, 0.0]))
    assert float(result["nDCGRendle2020"][0]) == pytest.approx(
        1 / torch.log2(torch.tensor(3.0)).item()
    )
    assert float(result["nDCGRendle2020"][1]) == 0
    assert float(result["ItemCoverage"]) == 2
    assert float(result["UserCoverage"]) == 1
    torch.testing.assert_close(result["NumRetrieved"], torch.tensor([2.0, 0.0]))
