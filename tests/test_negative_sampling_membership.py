"""Unsorted CSR rows must not turn positives into training negatives."""

import numpy as np
from scipy.sparse import csr_matrix
from warprec.data.entities.train_structures.interaction_structures import (
    NegativeSampler,
)


def test_unsorted_csr_never_returns_seen_items():
    """CSR permits unsorted indices; their order must not change exclusion."""
    matrix = csr_matrix((np.ones(2), [2, 0], [0, 2]), shape=(1, 4))
    assert not matrix.has_sorted_indices
    sampler = NegativeSampler(matrix, niid=4, seed=4)

    drawn = {sampler.sample(0) for _ in range(30)}

    assert drawn == {1, 3}
    np.testing.assert_array_equal(matrix.indices, [2, 0])
