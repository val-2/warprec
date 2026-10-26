"""Graph modules retain the standard PyTorch mode-switch contract."""

import pytest
import torch
from warprec.data.dataset import Dataset
from conftest import make_model


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
