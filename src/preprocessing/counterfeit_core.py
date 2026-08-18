"""
counterfeit_core.py
Pseudo-rating engine: converts raw implicit signals into usable relevance scores.
Supports 5 methods: weighted, pca, lasso, ridge, borda.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from tqdm import tqdm


_DEFAULT_WEIGHTS = {
    "purchase_count":  5.0,
    "cart_count":      3.0,
    "dwell_seconds":   0.001,
    "click_count":     1.0,
    "view_count":      0.2,
}

RATING_METHODS = ["weighted", "pca", "lasso", "ridge", "borda"]


class CounterFeitCore:
    """
    Pseudo-rating engine for implicit interaction data.

    Usage
    -----
    >>> engine = CounterFeitCore(method="weighted")
    >>> rated_df = engine.build_pseudo_ratings(interactions_df)
    """

    def __init__(self, method: str = "weighted", n_components: int = 1):
        if method not in RATING_METHODS:
            raise ValueError(f"method must be one of {RATING_METHODS}, got '{method}'")
        self.method = method
        self.n_components = n_components
        self._conn = duckdb.connect(":memory:")

    # ------------------------------------------------------------------
    def build_pseudo_ratings(
        self,
        interactions_df: pd.DataFrame,
        weight_config: dict | None = None,
    ) -> pd.DataFrame:
        """
        Build pseudo-rating scores from implicit behavioral signals.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Must contain [user_id, item_id] and at least one signal column.
        weight_config : dict | None
            Signal → weight mapping. Defaults to _DEFAULT_WEIGHTS.

        Returns
        -------
        pd.DataFrame  columns: [user_id, item_id, pseudo_rating]
        """
        if weight_config is None:
            weight_config = _DEFAULT_WEIGHTS

        available = [c for c in weight_config if c in interactions_df.columns]
        if not available:
            raise ValueError("interactions_df contains none of the expected signal columns.")

        dispatch = {
            "weighted": self._weighted,
            "pca":      self._pca,
            "lasso":    self._lasso,
            "ridge":    self._ridge,
            "borda":    self._borda,
        }
        with tqdm(total=1, desc=f"CounterFeitCore [{self.method}]", colour="#05ad46") as bar:
            result = dispatch[self.method](interactions_df, available, weight_config)
            bar.update(1)
        return result

    # ------------------------------------------------------------------  private

    def _weighted(self, df, cols, weights) -> pd.DataFrame:
        self._conn.register("interactions", df)
        score_expr = " + ".join(
            [f"COALESCE({c}, 0.0) * {weights[c]}" for c in cols]
        )
        sql = f"""
        WITH raw AS (
            SELECT user_id, item_id,
                   {score_expr} AS raw_score
            FROM interactions
        ),
        normed AS (
            SELECT user_id, item_id, raw_score,
                   (raw_score - MIN(raw_score) OVER()) /
                   NULLIF(MAX(raw_score) OVER() - MIN(raw_score) OVER(), 0) AS ns
            FROM raw
        )
        SELECT user_id, item_id,
               ROUND(COALESCE(ns, 0.0) * 4.0 + 1.0, 4) AS pseudo_rating
        FROM normed
        """
        return self._conn.execute(sql).fetchdf()

    def _pca(self, df, cols, _) -> pd.DataFrame:
        X = df[cols].fillna(0).values.astype(np.float32)
        pca = PCA(n_components=min(self.n_components, X.shape[1]))
        scores = pca.fit_transform(X)[:, 0]
        return self._normalise_scores(df, scores)

    def _lasso(self, df, cols, weights) -> pd.DataFrame:
        X = df[cols].fillna(0).values
        y = X @ np.array([weights[c] for c in cols])
        m = Lasso(alpha=0.01, max_iter=1000, fit_intercept=False)
        m.fit(X, y)
        return self._normalise_scores(df, m.predict(X))

    def _ridge(self, df, cols, weights) -> pd.DataFrame:
        X = df[cols].fillna(0).values
        y = X @ np.array([weights[c] for c in cols])
        m = Ridge(alpha=1.0, fit_intercept=False)
        m.fit(X, y)
        return self._normalise_scores(df, m.predict(X))

    def _borda(self, df, cols, _) -> pd.DataFrame:
        ranks = df[cols].fillna(0).rank(method="average")
        scores = ranks.mean(axis=1).values
        return self._normalise_scores(df, scores)

    @staticmethod
    def _normalise_scores(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        mn, mx = scores.min(), scores.max()
        norm = (scores - mn) / (mx - mn + 1e-9)
        return pd.DataFrame({
            "user_id":       df["user_id"].values,
            "item_id":       df["item_id"].values,
            "pseudo_rating": np.round(norm * 4.0 + 1.0, 4),
        })
