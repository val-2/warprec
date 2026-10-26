from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("UserCoverage")
class UserCoverage(TopKMetric):
    """The UserCoverage@k metric counts the number of users
       that received at least one recommendation.

    Attributes:
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_VALUES,
    }

    users: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(k, dist_sync_on_step)
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        users = kwargs.get("valid_users")

        values = self.block(kwargs, f"top_{self.k}_values", preds)
        # A relevant test item alone does not mean a list was produced.
        self.users += ((users > 0) & torch.isfinite(values).any(dim=1)).sum()

    def compute(self):
        return {self.name: float(self.users.item())}
