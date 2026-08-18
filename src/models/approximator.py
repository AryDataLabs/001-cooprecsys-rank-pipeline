"""
approximator.py — RAM-aware approximate nearest-neighbour scoring.
Falls back to numpy dot-product when psutil threshold is not met.
"""
import numpy as np
import psutil


class MemoryAwareApproximator:
    """
    Selects between exact (full dot-product) and approximate (FAISS)
    scoring based on available RAM at inference time.
    """

    def __init__(self, threshold_gb: float = 4.0):
        self.threshold_gb = threshold_gb
        self._faiss_index = None

    @property
    def _has_enough_ram(self) -> bool:
        return psutil.virtual_memory().available / 1e9 >= self.threshold_gb

    def build_index(self, item_factors: np.ndarray) -> None:
        """Build FAISS flat IP index from item embeddings."""
        try:
            import faiss
            d = item_factors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(d)
            self._faiss_index.add(item_factors.astype(np.float32))
            print(f"[Approximator] FAISS index built: {item_factors.shape[0]} items, d={d}")
        except ImportError:
            print("[Approximator] faiss not available; using numpy fallback")

    def score(
        self,
        user_vector: np.ndarray,
        item_factors: np.ndarray,
        top_k: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (top_k_item_indices, top_k_scores).
        Uses FAISS when available and RAM permits, else numpy.
        """
        if self._faiss_index is not None and self._has_enough_ram:
            q = user_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self._faiss_index.search(q, top_k)
            return indices[0], scores[0]

        # numpy fallback
        scores = item_factors @ user_vector
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx, scores[top_idx]
