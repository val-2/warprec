"""Training negatives must be unseen, and impossible draws must fail promptly."""

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from warprec.data.entities.train_structures.interaction_structures import (
    NegativeSampler,
)
from warprec.errors import DataError


def test_unsorted_csr_never_returns_seen_items():
    """CSR permits unsorted indices; their order must not change exclusion."""
    matrix = csr_matrix((np.ones(2), [2, 0], [0, 2]), shape=(1, 4))
    assert not matrix.has_sorted_indices
    sampler = NegativeSampler(matrix, niid=4, seed=4)

    drawn = {sampler.sample(0) for _ in range(30)}

    assert drawn == {1, 3}
    np.testing.assert_array_equal(matrix.indices, [2, 0])


class _NoDraw:
    """An impossible distribution must be rejected before consuming randomness."""

    def integers(self, *args: Any, **kwargs: Any) -> int:
        """Fail rather than hanging a regression on endless rejected samples."""
        pytest.fail("no unseen item is drawable, but sampling started")

    def random(self) -> float:
        """Fail rather than hanging a regression on endless rejected samples."""
        pytest.fail("no unseen item is drawable, but sampling started")


@pytest.mark.parametrize(
    "strategy,counts", [("uniform", [1.0, 1.0]), ("popularity", [1.0, 0.0])]
)
def test_training_rejects_an_empty_negative_support(strategy: str, counts: list[float]):
    """Popularity can be exhausted even when an unseen cold item exists."""
    sampler = NegativeSampler(csr_matrix([counts]), niid=2, strategy=strategy)
    sampler.rng = _NoDraw()

    with pytest.raises(DataError, match="negative|unseen"):
        sampler.sample(0)
