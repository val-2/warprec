"""Regression checks against the released PopDCL model and sampling protocol."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
import torch
from torch.nn import functional as F

from warprec.recommenders.collaborative_filtering_recommender.graph_based.popdcl import PopDCL


def model_and_matrix():
    matrix = sparse.csr_matrix((np.arange(1, 9),
        ([0, 0, 0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 3, 0, 4, 2])), shape=(4, 5))
    model = PopDCL(dict(embedding_size=4, n_layers=2, temperature=.27,
        reg_weight=1e-5, learning_rate=.001, batch_size=3, epochs=2),
        dict(n_users=4, n_items=5), SimpleNamespace(get_sparse=lambda: matrix), seed=42).double()
    return model, matrix


def test_binary_graph_last_layer_prior_and_scoring():
    model, original = model_and_matrix()
    binary = original.astype(float).copy(); binary.data.fill(1.)
    adjacency = sparse.bmat([[None, binary], [binary.T, None]]).toarray()
    inverse = adjacency.sum(1)**-.5
    graph = torch.tensor(inverse[:, None]*adjacency*inverse[None, :], dtype=torch.float64)
    initial = torch.cat([model.user_embedding.weight, model.item_embedding.weight[:-1]])
    expected = graph@graph@initial
    u, i = model.forward()
    torch.testing.assert_close(torch.cat([u, i[:-1]]), expected, rtol=2e-7, atol=1e-9)
    counts = np.asarray(binary.sum(0)).ravel()
    pop_sum = np.asarray(binary@counts).ravel()
    torch.testing.assert_close(model.user_pop_sum, torch.tensor(pop_sum))
    torch.testing.assert_close(model.omega_plus, torch.tensor((pop_sum-1)/7), rtol=1e-7, atol=1e-8)
    assert original.data.tolist() == list(range(1, 9))
    torch.testing.assert_close(model.predict(torch.arange(4)), u@i[:-1].T)
    assert not torch.allclose(model.predict(torch.arange(4)), F.normalize(u, dim=1)@F.normalize(i[:-1], dim=1).T)


def test_loss_and_gradients_match_released_formula():
    # SY1706203/PopDCL 5263b93, model.py PopDCL_LOSS_batch.forward:
    # M- uses the original score, then the negative is clamped before exp.
    model, _ = model_and_matrix()
    u = torch.tensor([[.2, -.3], [.5, .7], [-.4, .2]], dtype=torch.float64, requires_grad=True)
    i = torch.tensor([[.6, .1], [-.8, -.3], [.2, .9]], dtype=torch.float64, requires_grad=True)
    user, item = torch.tensor([0, 1, 2]), torch.tensor([0, 3, 4])
    un, inn = F.normalize(u, dim=1), F.normalize(i, dim=1)
    actual = model._compute_popdcl_loss(un, inn, user, item)
    scores = un@inn.T
    negatives = scores[~torch.eye(3, dtype=torch.bool)].view(3, 2)
    mplus = torch.sigmoid(negatives.mean(1)*model.item_pop[item]/model.user_pop_sum[user])
    numerator = torch.exp((scores.diag()-mplus)/model.temperature)
    omega = model.omega_plus[user, None]
    minus = omega/(1-omega)*torch.exp((scores.diag()[:, None]-negatives)/model.temperature)
    clamped = negatives.clamp_min(1e-7)
    beta = 1-minus/clamped
    denominator = numerator + torch.exp(beta*clamped/model.temperature).sum(1)
    reference = -torch.log(numerator/denominator).mean()
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    for a, b in zip(torch.autograd.grad(actual, (u, i), retain_graph=True),
                    torch.autograd.grad(reference, (u, i))):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


def test_sampler_weights_users_and_drops_partial_batch():
    model, matrix = model_and_matrix()
    loader = model.get_dataloader(SimpleNamespace(get_sparse=lambda: matrix), None, shuffle=False)
    assert len(loader.dataset) == matrix.nnz and loader.drop_last
    users = [loader.dataset[x][0] for x in range(len(loader.dataset))]
    assert users == [0, 1, 2, 3, 0, 1, 2, 3]
    random.seed(42); a = list(loader)
    random.seed(42); b = list(loader)
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            assert torch.equal(x, y)
        for user, item in zip(*left):
            assert matrix[int(user), int(item)] > 0


def test_penalty_checkpoint_and_legacy_rejection():
    model, _ = model_and_matrix()
    user, item = torch.tensor([0, 1, 2]), torch.tensor([0, 3, 4])
    u, i = model.forward()
    reference = model._compute_popdcl_loss(F.normalize(u[user], dim=1), F.normalize(i[item], dim=1), user, item)
    reference += model.reg_weight*.5/3*(model.user_embedding(user).square().sum()+model.item_embedding(item).square().sum())
    torch.testing.assert_close(model.training_step((user, item), 0), reference)
    restored, _ = model_and_matrix(); restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.predict(user), model.predict(user))
    legacy = dict(model.state_dict()); del legacy['implementation_version']
    with pytest.raises(RuntimeError, match='implementation_version'):
        restored.load_state_dict(legacy)
    checkpoint = model.get_state()
    torch.testing.assert_close(PopDCL.from_checkpoint(checkpoint).predict(user), model.predict(user))
    checkpoint['state_dict'] = legacy
    with pytest.raises(RuntimeError, match='implementation_version'):
        PopDCL.from_checkpoint(checkpoint, strict=False)
