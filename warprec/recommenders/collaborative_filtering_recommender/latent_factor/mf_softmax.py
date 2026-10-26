from typing import Any

from torch import Tensor
from torch.nn import functional as F

from warprec.recommenders.collaborative_filtering_recommender.latent_factor.bpr import BPR
from warprec.utils.registry import model_registry


@model_registry.register(name="MFSoftmax")
class MFSoftmax(BPR):
    """Dot-product matrix factorization with full-catalog cross entropy.

    Shares BPR's embeddings, initialization, optimizer, prediction and triple
    loader. The positive item is the class target; every real item participates
    in the softmax denominator, including other observed training items. The
    padding row is excluded. Training temperature is one, with no normalization
    or sampling correction. Memory for logits grows with batch size * catalog.

    The sampled negative contributes only to the same EmbLoss penalty used by
    BPR, allowing a comparison that changes only the data-fitting objective.
    This is full softmax, not sampled or in-batch softmax.
    """

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, pos_item, neg_item = batch
        logits = self.predict(user)
        data_loss = F.cross_entropy(logits, pos_item)
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )
        loss = data_loss + reg_loss
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
