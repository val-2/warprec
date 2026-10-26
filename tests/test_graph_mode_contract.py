"""Graph modules retain the standard PyTorch mode-switch contract."""

import pytest
from warprec.data.dataset import Dataset
from conftest import make_model


@pytest.mark.parametrize("mode", [True, False])
def test_graph_mode_switch_returns_the_model(dataset: Dataset, mode: bool):
    """Graph models retain the standard nn.Module chaining contract."""
    model = make_model("LightGCN", dataset)
    assert model.train(mode) is model
    assert model.training is mode
    assert model.eval() is model
