# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
_cy_predict.pyx
Cython-accelerated dot-product scoring for ALS recommendation.
Computes top-K items for every user via batched matrix-vector products.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t


def cy_top_k_scores(
    cnp.ndarray[DTYPE_t, ndim=2] user_factors,
    cnp.ndarray[DTYPE_t, ndim=2] item_factors,
    int top_k,
    cnp.ndarray[cnp.int32_t, ndim=1] exclude_items = None,
):
    """
    Compute top-K item indices for each user via Cython-accelerated dot product.

    Parameters
    ----------
    user_factors : float32 ndarray (n_users, n_factors)
    item_factors : float32 ndarray (n_items, n_factors)
    top_k        : int
    exclude_items: int32 ndarray of item indices to mask (optional)

    Returns
    -------
    top_k_indices : int32 ndarray (n_users, top_k)
    top_k_scores  : float32 ndarray (n_users, top_k)
    """
    cdef int n_users = user_factors.shape[0]
    cdef int n_items = item_factors.shape[0]
    cdef int n_factors = user_factors.shape[1]
    cdef int u, i, k, f
    cdef DTYPE_t s

    cdef cnp.ndarray[DTYPE_t, ndim=1] scores = np.empty(n_items, dtype=DTYPE)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out_idx = np.empty((n_users, top_k), dtype=np.int32)
    cdef cnp.ndarray[DTYPE_t, ndim=2] out_scr = np.empty((n_users, top_k), dtype=DTYPE)

    # Build item factor matrix view
    cdef DTYPE_t[:, :] U = user_factors
    cdef DTYPE_t[:, :] V = item_factors
    cdef DTYPE_t[:] S = scores

    for u in range(n_users):
        # dot product: user u with every item
        for i in range(n_items):
            s = 0.0
            for f in range(n_factors):
                s += U[u, f] * V[i, f]
            S[i] = s

        # mask excluded items
        if exclude_items is not None:
            for k in range(exclude_items.shape[0]):
                S[exclude_items[k]] = -1e9

        # partial sort: argsort descending, take top_k
        ranked = np.argsort(scores)[::-1][:top_k]
        out_idx[u] = ranked.astype(np.int32)
        out_scr[u] = scores[ranked]

    return out_idx, out_scr


def cy_cosine_similarity_rows(
    cnp.ndarray[DTYPE_t, ndim=2] A,
    cnp.ndarray[DTYPE_t, ndim=2] B,
):
    """
    Row-wise cosine similarity: sim[i] = dot(A[i], B[i]) / (|A[i]| * |B[i]|).
    Used for item-item similarity in Item-CF.
    """
    cdef int n = A.shape[0]
    cdef int d = A.shape[1]
    cdef int i, j
    cdef DTYPE_t dot, na, nb
    cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)

    for i in range(n):
        dot = 0.0
        na  = 0.0
        nb  = 0.0
        for j in range(d):
            dot += A[i, j] * B[i, j]
            na  += A[i, j] * A[i, j]
            nb  += B[i, j] * B[i, j]
        result[i] = dot / (sqrt(na) * sqrt(nb) + 1e-9)
    return result
