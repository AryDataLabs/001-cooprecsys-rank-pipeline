"""
trainer.py — AryColBringTrainer: ALS-WR collaborative filtering trainer.
"""
from __future__ import annotations
import time, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.sparse import csr_matrix
from tqdm import tqdm
import implicit

from .base import AbstractRecommenderBase
from cooprecsys.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k


class AryColBringTrainer(AbstractRecommenderBase):
    """
    ALS-WR collaborative filtering trainer wrapping the `implicit` library
    with cooprecsys-specific RAM-aware batching and MLflow integration.

    Parameters
    ----------
    n_factors      : int   — latent dimension
    n_iterations   : int   — ALS iterations
    regularization : float — L2 regularisation
    alpha          : float — confidence weight multiplier
    random_state   : int
    """

    def __init__(
        self,
        n_factors: int = 128,
        n_iterations: int = 30,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        self._model: implicit.als.AlternatingLeastSquares | None = None
        self._interaction_matrix: csr_matrix | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        interaction_matrix: csr_matrix,
        show_progress: bool = True,
        tqdm_colour: str = "#05ad46",
    ) -> "AryColBringTrainer":
        """
        Fit ALS model on the user-item interaction matrix.

        Parameters
        ----------
        interaction_matrix : csr_matrix  (n_users, n_items)
        show_progress      : bool
        tqdm_colour        : str    hex colour for tqdm bars
        """
        self._interaction_matrix = interaction_matrix

        print(f"[AryColBringTrainer] RAM available: "
              f"{psutil.virtual_memory().available / 1e9:.1f} GB")

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.n_iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            calculate_training_loss=True,
            use_gpu=False,
        )

        # implicit expects item-user matrix (items as rows)
        item_user = (interaction_matrix.T * self.alpha).tocsr()

        t0 = time.time()
        self._model.fit(item_user, show_progress=show_progress)
        elapsed = time.time() - t0
        print(f"[AryColBringTrainer] Training complete in {elapsed:.1f}s")
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, user_id: int, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Return (item_indices, scores) for a single user."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        ids, scores = self._model.recommend(
            user_id,
            self._interaction_matrix[user_id],
            N=top_k,
            filter_already_liked_items=True,
        )
        return np.array(ids), np.array(scores)

    # ------------------------------------------------------------------ evaluate
    def evaluate(
        self,
        holdout_matrix: csr_matrix,
        k_values: list[int] = [5, 10, 20],
        max_users: int = 5000,
        tqdm_colour: str = "#05ad46",
    ) -> dict[str, float]:
        """
        Compute offline ranking metrics on a holdout matrix.

        Parameters
        ----------
        holdout_matrix : csr_matrix  (n_users, n_items)  ground truth
        k_values       : list[int]
        max_users      : int   cap for evaluation speed
        """
        n_users = min(holdout_matrix.shape[0], max_users)
        results = {f"ndcg@{k}": [] for k in k_values}
        results.update({f"precision@{k}": [] for k in k_values})
        results.update({f"recall@{k}": [] for k in k_values})

        user_ids = np.arange(n_users)
        np.random.default_rng(42).shuffle(user_ids)

        for uid in tqdm(user_ids, desc="Evaluating", colour=tqdm_colour, leave=False):
            true_items = holdout_matrix[uid].indices
            if len(true_items) == 0:
                continue
            pred_items, _ = self.predict(uid, top_k=max(k_values))
            for k in k_values:
                results[f"ndcg@{k}"].append(ndcg_at_k(true_items, pred_items, k))
                results[f"precision@{k}"].append(precision_at_k(true_items, pred_items, k))
                results[f"recall@{k}"].append(recall_at_k(true_items, pred_items, k))

        return {key: float(np.mean(vals)) for key, vals in results.items()}

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "matrix": self._interaction_matrix}, f)
        print(f"[AryColBringTrainer] Saved to {path}")

    def load(self, path: str) -> "AryColBringTrainer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._interaction_matrix = data["matrix"]
        return self
