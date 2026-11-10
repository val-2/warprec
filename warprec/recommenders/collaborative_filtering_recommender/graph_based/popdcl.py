"""PopDCL aligned with SY1706203/PopDCL at 5263b93ea44c4750a6650bcfe4059cb4dbe5e637.

The released code uses the last graph layer, user-uniform positive sampling,
normal(.1) initialization, an adjusted prior and a negative-score floor. These
choices differ from parts of the paper and former WarpRec. Old checkpoints
require the former implementation; a strict load rejects their missing version.
"""
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from warprec.data.entities import Interactions, Sessions
from warprec.data.entities.common import seeded_dataloader
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import GraphRecommenderUtils
from warprec.recommenders.losses import EmbLoss
from warprec.utils.registry import model_registry


class _PopDCLPairs(Dataset):
    """N draws with near-uniform users and replacement within each history."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.users = np.flatnonzero(np.diff(matrix.indptr))

    def __len__(self):
        return self.matrix.nnz

    def __getitem__(self, index):
        user = int(self.users[index % len(self.users)])
        begin, end = self.matrix.indptr[user:user+2]
        # WarpRec seeds Python random in the model and in loader workers.
        return user, int(random.choice(self.matrix.indices[begin:end]))


@model_registry.register(name="PopDCL")
class PopDCL(GraphRecommenderUtils, IterativeRecommender):
    """Released PopDCL core, using WarpRec graph, optimizer and EmbLoss.

    reg_weight multiplies .5/B times squared initial user/positive norms.
    Training uses cosine; prediction exposes raw dot logits (the authors apply
    a monotone sigmoid after the dot). No correction is used at inference.
    """
    embedding_size: int
    n_layers: int
    temperature: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(self, params: dict, info: dict, interactions: Interactions,
                 *args: Any, seed: int = 42, **kwargs: Any):
        super().__init__(params, info, *args, seed=seed, **kwargs)
        if self.temperature <= 0:
            raise ValueError("PopDCL temperature must be positive")
        # Counts and graph edges are binary even for explicit-rating inputs.
        matrix = interactions.get_sparse().tocsr(copy=True)
        matrix.eliminate_zeros()
        matrix.data.fill(1.)
        total = matrix.nnz
        if total <= 1:
            raise ValueError("PopDCL requires at least two training interactions")
        user_pop = np.asarray(matrix.sum(axis=1)).ravel()
        item_pop = np.asarray(matrix.sum(axis=0)).ravel()
        pop_sum = np.asarray(matrix @ item_pop).ravel()
        omega = np.where(user_pop > 0, (pop_sum-1)/(total-1), 0.)
        if np.any(omega >= 1) or np.any(omega < 0):
            raise ValueError("PopDCL false-negative prior must be in [0, 1)")
        self.register_buffer("user_pop", torch.as_tensor(user_pop, dtype=torch.float32))
        self.register_buffer("item_pop", torch.tensor(np.r_[item_pop, 0.], dtype=torch.float32))
        self.register_buffer("user_pop_sum", torch.as_tensor(pop_sum, dtype=torch.float32))
        self.register_buffer("omega_plus", torch.as_tensor(omega, dtype=torch.float32))
        self.register_buffer("implementation_version", torch.tensor(2))
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items+1, self.embedding_size, padding_idx=self.n_items)
        nn.init.normal_(self.user_embedding.weight, std=.1)
        nn.init.normal_(self.item_embedding.weight, std=.1)
        with torch.no_grad():
            self.item_embedding.weight[-1].zero_()
        self.adj = self.get_adj_mat(matrix.tocoo(), self.n_users, self.n_items+1, normalize=True)
        self.reg_loss = EmbLoss()

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs: Any):
        matrix = interactions.get_sparse().tocsr(copy=True)
        matrix.eliminate_zeros()
        seed = kwargs.pop("seed", self.seed)
        shuffle = kwargs.pop("shuffle", True)
        kwargs.setdefault("drop_last", True)
        return seeded_dataloader(_PopDCLPairs(matrix), batch_size=self.batch_size,
                                 shuffle=shuffle, seed=seed, **kwargs)

    def forward(self) -> Tuple[Tensor, Tensor]:
        current = self.get_ego_embeddings(self.user_embedding, self.item_embedding)
        if self.adj.device() != current.device:
            self.adj = self.adj.to(current.device)
        for _ in range(self.n_layers):
            current = self.adj.matmul(current)
        return current.split([self.n_users, self.n_items+1])

    def _compute_popdcl_loss(self, user_emb: Tensor, item_emb_batch: Tensor,
                             user_idx: Tensor, pos_item_idx: Tensor) -> Tensor:
        scores = user_emb @ item_emb_batch.T
        positive = scores.diag()
        negative_mean = (scores.sum(1)-positive)/max(len(scores)-1, 1)
        p_false_pos = self.item_pop[pos_item_idx]/self.user_pop_sum[user_idx].clamp_min(1.)
        mplus = torch.sigmoid(p_false_pos*negative_mean)
        omega = self.omega_plus[user_idx, None]
        minus = omega/(1-omega)*torch.exp((positive[:, None]-scores)/self.temperature)
        # M- precedes the floor in the authors' code. Avoid their division by
        # an almost-zero score with this equivalent expression and gradient.
        negatives = (scores.clamp_min(1e-7)-minus)/self.temperature
        negatives = negatives.masked_fill(torch.eye(len(scores), device=scores.device, dtype=torch.bool), -torch.inf)
        pos = (positive-mplus)/self.temperature
        return (torch.logsumexp(torch.cat([pos[:, None], negatives], 1), 1)-pos).mean()

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, positive = batch[:2]
        all_users, all_items = self.forward()
        loss = self._compute_popdcl_loss(F.normalize(all_users[user], dim=1),
            F.normalize(all_items[positive], dim=1), user, positive)
        loss = loss + self.reg_weight*self.reg_loss(self.user_embedding(user), self.item_embedding(positive))
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict(self, user_indices: Tensor, *args: Any,
                item_indices: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        """Unnormalized last-layer dot products, excluding padding."""
        users, items = self.propagate_embeddings()
        if item_indices is None:
            return users[user_indices] @ items[:self.n_items].T
        return torch.einsum("bd,bsd->bs", users[user_indices], items[item_indices])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if int(state_dict.get("implementation_version", 0)) != 2:
            raise RuntimeError("PopDCL implementation_version mismatch; legacy weights require their original code")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_checkpoint(cls, checkpoint: Any, strict: bool = True, **kwargs: Any):
        # Base checkpoints may restore a whole module without calling __init__
        # or load_state_dict, so check this entry point too.
        if int(checkpoint["state_dict"].get("implementation_version", 0)) != 2:
            raise RuntimeError("PopDCL implementation_version mismatch; legacy weights require their original code")
        return super().from_checkpoint(checkpoint, strict=strict, **kwargs)
