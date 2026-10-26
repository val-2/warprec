"""Graph modules retain the standard PyTorch mode-switch contract."""

import torch
from warprec.data.dataset import Dataset
from conftest import make_model


def test_graph_dtype_conversion_discards_the_old_cache(dataset: Dataset):
    """A cached float32 prediction cannot survive converting the weights to float64."""
    model = make_model("LightGCN", dataset)
    # No propagation layers isolate cache conversion from sparse-matmul precision.
    model.n_layers = 0
    model.eval()
    users = torch.arange(3)
    with torch.inference_mode():
        assert model.predict(users).dtype == torch.float32
    model.double()
    with torch.inference_mode():
        assert model.predict(users).dtype == torch.float64
