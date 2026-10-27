"""Full-catalog loss and chunked epoch gradients on the unchanged graph encoder."""
from types import SimpleNamespace

import numpy as np
from scipy import sparse
import torch

from warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcn import LightGCN
from warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcn_softmax import LightGCNSoftmax
from warprec.utils.registry import model_registry, params_registry


def models():
    matrix = sparse.csr_matrix((np.ones(4), ([0,0,1,1], [0,1,1,2])), shape=(2,4))
    interactions = SimpleNamespace(get_sparse=lambda: matrix)
    params = dict(embedding_size=3, n_layers=2, reg_weight=.01,
                  batch_size=2, epochs=2, learning_rate=.03)
    info = dict(n_users=2,n_items=4)
    return [cls(params,info,interactions,seed=42).double() for cls in (LightGCN,LightGCNSoftmax)]


def test_full_softmax_encoder_penalty_and_padding():
    base, model = models()
    u,p,n = torch.tensor([0,1]),torch.tensor([0,2]),torch.tensor([3,0])
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[key])
    eu,ei = model.forward()
    logits = eu[u] @ ei[:4].T
    reference = (torch.logsumexp(logits,1)-logits[torch.arange(2),p]).mean()
    reference += model.reg_weight*model.reg_loss(model.user_embedding(u),model.item_embedding(p),model.item_embedding(n))
    torch.testing.assert_close(model.training_step((u,p,n),0),reference,rtol=1e-12,atol=1e-12)
    altered = ei.clone(); altered[4].fill_(1e9)
    torch.testing.assert_close(model.data_loss(eu[u],ei,p),model.data_loss(eu[u],altered,p))


def test_chunked_full_batch_gradient_matches_single_batch():
    _, model = models()
    u,p = torch.tensor([0,0,1,1]),torch.tensor([0,1,1,2])
    eu,ei = model.forward()
    expected = model.data_loss(eu[u],ei,p)
    expected.backward()
    gradient = [v.grad.clone() for v in model.parameters()]
    model.zero_grad()
    eu,ei = model.forward()
    live = torch.cat([eu,ei]); leaf = live.detach().requires_grad_(True)
    for begin in (0,2):
        loss = model.data_loss(leaf[u[begin:begin+2]],leaf[2:],p[begin:begin+2])*.5
        loss.backward()
    live.backward(leaf.grad)
    for expected, parameter in zip(gradient,model.parameters()):
        torch.testing.assert_close(parameter.grad,expected,rtol=1e-12,atol=1e-12)


def test_registry_and_checkpoint():
    _,model = models()
    assert model_registry.get_class("LightGCNSoftmax") is LightGCNSoftmax
    assert params_registry.get_class("LightGCNSoftmax") is not None
    restored = LightGCNSoftmax.from_checkpoint(checkpoint=model.get_state())
    torch.testing.assert_close(restored.predict(torch.tensor([0,1])),model.predict(torch.tensor([0,1])))
