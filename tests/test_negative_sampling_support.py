"""Impossible negative distributions must terminate before drawing."""

from typing import Any
import pytest
from scipy.sparse import csr_matrix
from warprec.data.entities.train_structures.interaction_structures import (
    NegativeSampler,
)
from warprec.errors import DataError


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
