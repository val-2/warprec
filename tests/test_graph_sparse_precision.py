"""Graph modules retain the standard PyTorch mode-switch contract."""

import pytest
import torch
from warprec.data.dataset import Dataset
from conftest import make_model


@pytest.mark.parametrize("score_before_conversion", [False, True])
def test_graph_predictions_follow_parameter_precision(
    dataset: Dataset, score_before_conversion: bool
):
    """Float64 scoring works both before and after populating a float32 cache."""
    model = make_model("LightGCN", dataset)
    model.eval()
    users = torch.arange(3)
    if score_before_conversion:
        with torch.inference_mode():
            model.predict(users)
    model.double()

    with torch.inference_mode():
        scores = model.predict(users)

    assert scores.dtype == torch.float64


def test_lightgcn_float64_loss_has_finite_gradients(dataset: Dataset):
    """The dtype correction also supports backpropagation through the graph."""
    model = make_model("LightGCN", dataset).double()
    batch = next(iter(model.get_dataloader(dataset.train_set, dataset.train_session)))

    loss = model.training_step(batch, 0)
    loss.backward()

    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)
    for embedding in (model.user_embedding, model.item_embedding):
        assert embedding.weight.grad is not None
        assert torch.isfinite(embedding.weight.grad).all()
