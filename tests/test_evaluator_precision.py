"""Device placement must preserve an explicitly selected metric precision."""

import math

import pytest
import torch
from scipy.sparse import csr_matrix

from warprec.evaluation.evaluator import Evaluator


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_device_placement_preserves_metric_precision(dtype):
    evaluator = Evaluator(
        ["nDCGRendle2020", "ARP", "APLT", "Gini"],
        [3],
        train_set=csr_matrix([[1, 1, 0, 0]]),
    )
    for metric in evaluator.metrics[3]:
        metric.set_dtype(dtype)
    # evaluate() resets and places metrics before processing the batches.
    for _ in range(2):
        evaluator.reset_metrics()
        evaluator.metrics_to("cpu")
        for metric in evaluator.metrics[3]:
            assert metric.dtype == dtype
        ndcg = evaluator.metrics[3][0]
        ndcg.update(
            preds=torch.tensor([[1.0, 0.0, 2.0, -torch.inf]]),
            ground=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            user_indices=torch.tensor([0]),
        )
        actual = ndcg.compute()["nDCGRendle2020"].item()
        if dtype == torch.float64:
            expected = math.fsum([1 / math.log2(3), 0.5]) / math.fsum(
                [1.0, 1 / math.log2(3)]
            )
            assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
