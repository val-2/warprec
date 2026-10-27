from typing import Any

from torch import Tensor
from torch.nn import functional as F

from warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcn import LightGCN
from warprec.utils.registry import model_registry


@model_registry.register(name="LightGCNSoftmax")
class LightGCNSoftmax(LightGCN):
    """LightGCN encoder with full-catalog cross entropy at temperature one.

    Propagation, initialization, dot-product inference and sampled base-embedding
    L2 are inherited unchanged. All real items, including training positives,
    enter the denominator; padding is excluded. The negative contributes only to
    EmbLoss. Logit memory scales with batch size times catalog size.
    """

    def data_loss(self, users: Tensor, items: Tensor, positives: Tensor) -> Tensor:
        """Mean full-softmax loss, reusable when accumulating an epoch's gradient.

        Inputs are propagated user vectors for the batch and all item vectors.
        This method has no regularization or optimizer step, allowing a caller
        to preserve a full-batch training schedule without another propagation.
        """
        return F.cross_entropy(users @ items[: self.n_items].T, positives)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, pos_item, neg_item = batch
        all_users, all_items = self.forward()
        data_loss = self.data_loss(all_users[user], all_items, pos_item)
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )
        loss = data_loss + reg_loss
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
