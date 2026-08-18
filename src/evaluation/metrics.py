"""
metrics.py — Standard information-retrieval ranking metrics.
"""
import numpy as np


def _dcg(relevances: np.ndarray, k: int) -> float:
    r = relevances[:k]
    if r.size == 0:
        return 0.0
    positions = np.arange(2, r.size + 2)
    return float(np.sum((2.0 ** r - 1.0) / np.log2(positions)))


def ndcg_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K."""
    true_set = set(true_items)
    gains = np.array([1.0 if p in true_set else 0.0 for p in pred_items[:k]])
    ideal = np.ones(min(len(true_set), k))
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(gains, k) / idcg


def precision_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
    """Precision @ K."""
    if k == 0:
        return 0.0
    true_set = set(true_items)
    hits = sum(1 for p in pred_items[:k] if p in true_set)
    return hits / k


def recall_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
    """Recall @ K."""
    true_set = set(true_items)
    if not true_set:
        return 0.0
    hits = sum(1 for p in pred_items[:k] if p in true_set)
    return hits / len(true_set)


def mean_reciprocal_rank(true_items: np.ndarray, pred_items: np.ndarray) -> float:
    """Mean Reciprocal Rank (MRR)."""
    true_set = set(true_items)
    for rank, item in enumerate(pred_items, start=1):
        if item in true_set:
            return 1.0 / rank
    return 0.0


def mean_average_precision(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
    """Mean Average Precision @ K."""
    true_set = set(true_items)
    hits, sum_prec = 0, 0.0
    for i, p in enumerate(pred_items[:k], start=1):
        if p in true_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(true_set), k) if true_set else 0.0
