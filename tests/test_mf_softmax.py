"""Full-catalog MF changes the ranking objective while keeping BPR's encoder."""
import torch
from torch.nn import functional as F

from warprec.recommenders.collaborative_filtering_recommender.latent_factor.bpr import BPR
from warprec.recommenders.collaborative_filtering_recommender.latent_factor.mf_softmax import MFSoftmax
from warprec.utils.registry import model_registry, params_registry


def models(reg=0.03):
    params = dict(embedding_size=3, reg_weight=reg, batch_size=2,
                  epochs=2, learning_rate=0.001)
    info = dict(n_users=2, n_items=4)
    return BPR(params, info, seed=42), MFSoftmax(params, info, seed=42)


def test_full_softmax_value_gradient_and_padding():
    bpr, model = models()
    assert torch.equal(bpr.user_embedding.weight, model.user_embedding.weight)
    assert torch.equal(bpr.item_embedding.weight, model.item_embedding.weight)
    u, p, n = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 0])
    # A huge padding vector must never enter the catalog denominator.
    with torch.no_grad():
        model.item_embedding.weight[4].fill_(1e6)
    logits = model.user_embedding(u) @ model.item_embedding.weight[:4].T
    reference = (torch.logsumexp(logits, 1) - logits[torch.arange(2), p]).mean()
    reference += model.reg_weight * model.reg_loss(
        model.user_embedding(u), model.item_embedding(p), model.item_embedding(n))
    actual = model.training_step((u, p, n), 0)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert torch.isfinite(model.item_embedding.weight.grad).all()
    assert torch.equal(model.item_embedding.weight.grad[4], torch.zeros(3))
    assert model.item_embedding.weight.grad[0].abs().sum() > 0


def test_negative_affects_only_the_shared_regularizer():
    bpr, model = models(reg=0.)
    u, p = torch.tensor([0, 1]), torch.tensor([1, 2])
    n1, n2 = torch.tensor([0, 0]), torch.tensor([3, 3])
    torch.testing.assert_close(model.training_step((u, p, n1), 0),
                               model.training_step((u, p, n2), 0))
    bpr.reg_weight = model.reg_weight = 0.03
    batch = (u, p, n1)
    logits = model.predict(u)
    sm_penalty = model.training_step(batch, 0) - F.cross_entropy(logits, p)
    bpr_penalty = bpr.training_step(batch, 0) - bpr.bpr_loss(bpr(u, p), bpr(u, n1))
    torch.testing.assert_close(sm_penalty, bpr_penalty, rtol=1e-4, atol=1e-7)


def test_registry_config_and_checkpoint():
    _, model = models()
    assert model_registry.get_class("MFSoftmax") is MFSoftmax
    config = params_registry.get_class("MFSoftmax")(**dict(
        embedding_size=[3], reg_weight=[.03], batch_size=[2], epochs=[2], learning_rate=[.001]))
    assert config.embedding_size[-1] == 3
    restored = MFSoftmax.from_checkpoint(checkpoint=model.get_state())
    torch.testing.assert_close(model.predict(torch.tensor([0, 1])),
                               restored.predict(torch.tensor([0, 1])))
