"""
base.py — Abstract base class for all cooprecsys recommenders.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix


class AbstractRecommenderBase(ABC):
    """Base interface every cooprecsys recommender must implement."""

    @abstractmethod
    def fit(self, interaction_matrix: csr_matrix, **kwargs) -> "AbstractRecommenderBase":
        ...

    @abstractmethod
    def predict(self, user_id: int, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Returns (item_indices, scores) arrays of length top_k."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> "AbstractRecommenderBase":
        ...
