from typing import Any, Optional

import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from torch import Tensor
from scipy.sparse import coo_matrix

from warprec.errors import DataError, DataNotFoundError, ModelError
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="ProxyRecommender")
class ProxyRecommender(Recommender):
    """Implementation of a ProxyRecommender, used
    to evaluate a recommendation file from other frameworks.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        recommendation_file (str): Path to the recommendation file.
        separator (str): Separator of the recommendation file.
        header (bool): Wether or not the recommendation file has an header.
        missing_policy (str): 'zero' preserves upstream score completion;
            'unavailable' evaluates only explicitly listed pairs.

    Raises:
        ModelError: If the item and user mappings or the number of items and
            users are not provided.
        DataError: If the recommendation file is empty or malformed.
        DataNotFoundError: If the recommendation file does not exist.
        RuntimeError: If an error occurs while reading the recommendation file.
    """

    recommendation_file: str
    separator: str
    header: bool
    missing_policy: str = "zero"

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)
        if self.missing_policy not in ("zero", "unavailable"):
            raise ModelError("missing_policy must be 'zero' or 'unavailable'")

        imap: dict = info.get("item_mapping", None)
        umap: dict = info.get("user_mapping", None)
        if any(x is None or x == {} for x in [imap, umap]):
            raise ModelError(
                "Item and user mapping must be provided to correctly initialize the model."
            )

        try:
            recommendation_df: DataFrame
            if self.header:
                recommendation_df = pd.read_csv(
                    self.recommendation_file,
                    sep=self.separator,
                    dtype={"user_id": int, "item_id": int, "rating": float},
                    usecols=["user_id", "item_id", "rating"],
                )
            else:
                recommendation_df = pd.read_csv(
                    self.recommendation_file, sep=self.separator, header=None
                )
                recommendation_df = recommendation_df.iloc[:, :3]
                recommendation_df.columns = ["user_id", "item_id", "rating"]

            users = recommendation_df["user_id"].map(umap).values
            items = recommendation_df["item_id"].map(imap).values
            ratings = recommendation_df["rating"].values

            # Compute invalid values mask for faster filtering
            mask = ~np.isnan(users) & ~np.isnan(items) & np.isfinite(ratings)

            # Filter out invalid values
            users = users[mask]
            items = items[mask]
            ratings = ratings[mask]

            self.predictions_sparse = coo_matrix(
                (ratings, (users, items)),
                shape=(self.n_users, self.n_items),
                dtype=np.float32,
            ).tocsr()
        except FileNotFoundError as exc:
            raise DataNotFoundError(
                f"Recommendation file {self.recommendation_file} not found."
            ) from exc
        except pd.errors.EmptyDataError as exc:
            raise DataError(
                f"Recommendation file {self.recommendation_file} is empty."
            ) from exc
        except pd.errors.ParserError as exc:
            raise DataError(
                f"Recommendation file {self.recommendation_file} is malformed."
            ) from exc
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while reading the recommendation file: {e}"
            ) from e

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction used a mocked model learned from a recommendation file.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Access the scores from the recommendation file
        batch = self.predictions_sparse[user_indices.cpu().numpy()].tocoo()
        # Preserve upstream completion by default. Exact partial-list import
        # is an explicit policy, independent of the sign of listed scores.
        fill = 0.0 if self.missing_policy == "zero" else -np.inf
        predictions_numpy = np.full(batch.shape, fill, dtype=batch.dtype)
        predictions_numpy[batch.row, batch.col] = batch.data
        predictions = torch.from_numpy(predictions_numpy)

        # Return full or sampled predictions
        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        indices = item_indices.to(predictions.device)
        valid = (indices >= 0) & (indices < self.n_items)
        sampled = predictions.gather(1, indices.clamp(min=0, max=self.n_items - 1))
        return sampled.masked_fill(~valid, -torch.inf)
