"""SASRec trains and predicts over the same real-item catalogue."""
import torch
from torch.nn import functional as F

from warprec.recommenders.sequential_recommender.sasrec import SASRec


def test_softmax_excludes_padding_loss_and_gradients():
    params = dict(embedding_size=8, n_layers=1, n_heads=2, inner_size=16,
                  dropout_prob=0., attn_dropout_prob=0., reg_weight=0.,
                  weight_decay=0., batch_size=2, epochs=1, learning_rate=.001,
                  neg_samples=0, max_seq_len=3)
    model = SASRec(params, dict(n_users=2, n_items=5), seed=42)
    model.log = lambda *args, **kwargs: None
    model.eval()
    batch = (torch.tensor([[0, 1, 5], [2, 3, 4]]),
             torch.tensor([2, 3]), torch.tensor([2, 0]))
    expected = F.cross_entropy(model.predict(batch[0], batch[1]), batch[2])
    expected_grad = torch.autograd.grad(expected, model.item_embedding.weight)[0]
    actual = model.training_step(batch, 0)
    actual_grad = torch.autograd.grad(actual, model.item_embedding.weight)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.count_nonzero(actual_grad[5]) == 0
