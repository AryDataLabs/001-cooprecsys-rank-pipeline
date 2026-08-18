"""
cf_preprocess.py
Builds a sparse user-item interaction matrix from a ratings DataFrame.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def build_interaction_matrix(
    rated_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "pseudo_rating",
) -> tuple[csr_matrix, LabelEncoder, LabelEncoder]:
    """
    Convert a long-format interaction DataFrame into a sparse CSR matrix.

    Parameters
    ----------
    rated_df   : pd.DataFrame  — columns [user_col, item_col, rating_col]
    user_col   : str
    item_col   : str
    rating_col : str

    Returns
    -------
    interaction_matrix : scipy.sparse.csr_matrix  shape (n_users, n_items)
    user_encoder       : sklearn.LabelEncoder
    item_encoder       : sklearn.LabelEncoder
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    user_idx = user_enc.fit_transform(rated_df[user_col].values)
    item_idx = item_enc.fit_transform(rated_df[item_col].values)
    ratings  = rated_df[rating_col].astype(np.float32).values

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    matrix = csr_matrix(
        (ratings, (user_idx, item_idx)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    print(f"[cf_preprocess] Matrix: {n_users} users × {n_items} items | "
          f"density={matrix.nnz / (n_users * n_items):.5%}")
    return matrix, user_enc, item_enc


def temporal_train_test_split(
    rated_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/test split: last `test_ratio` of interactions → test.
    """
    rated_df = rated_df.sort_values(timestamp_col)
    split_idx = int(len(rated_df) * (1 - test_ratio))
    return rated_df.iloc[:split_idx].copy(), rated_df.iloc[split_idx:].copy()
