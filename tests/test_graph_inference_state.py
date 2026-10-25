"""Graph inference must reflect current weights, precision and module mode."""

import pytest
import torch

from warprec.data.dataset import Dataset

from conftest import make_model


@pytest.mark.parametrize("mode", [True, False])
def test_graph_mode_switch_returns_the_model(dataset: Dataset, mode: bool):
    """Graph models retain the standard nn.Module chaining contract."""
    model = make_model("LightGCN", dataset)
    assert model.train(mode) is model
    assert model.training is mode
    assert model.eval() is model


@pytest.mark.parametrize("nested", [False, True])
def test_graph_loading_weights_invalidates_propagated_embeddings(
    dataset: Dataset, nested: bool
):
    """Loading zero weights after scoring must not return old nonzero scores."""
    model = make_model("LightGCN", dataset)
    model.eval()
    users = torch.arange(3)
    with torch.inference_mode():
        before = model.predict(users)
    assert before.abs().max() > 0

    target = torch.nn.Sequential(model) if nested else model
    zeros = {key: torch.zeros_like(value) for key, value in target.state_dict().items()}
    target.load_state_dict(zeros)
    with torch.inference_mode():
        after = model.predict(users)

    torch.testing.assert_close(after, torch.zeros_like(after), rtol=0, atol=0)


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
