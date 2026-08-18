"""
predictor.py — AryColBringPredictor: online inference from trained ALS model.
"""
from __future__ import annotations
import numpy as np
import psutil
from sklearn.preprocessing import LabelEncoder
from .trainer import AryColBringTrainer


class AryColBringPredictor:
    """
    Wraps a trained AryColBringTrainer for production inference.
    Handles user/item encoding and memory-aware batch sizing.
    """

    def __init__(
        self,
        model: AryColBringTrainer,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        memory_threshold_gb: float = 8.0,
    ):
        self.model = model
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.memory_threshold_gb = memory_threshold_gb

    def recommend(self, user_id_raw: str | int, top_k: int = 20) -> list[dict]:
        """
        Return top-K recommendations for a raw user ID.

        Returns
        -------
        list of {"item_id": str, "score": float, "rank": int}
        """
        try:
            uid_enc = self.user_encoder.transform([user_id_raw])[0]
        except ValueError:
            return []  # unknown user → cold start handled upstream

        item_indices, scores = self.model.predict(uid_enc, top_k=top_k)
        item_ids = self.item_encoder.inverse_transform(item_indices)
        return [
            {"item_id": str(iid), "score": float(s), "rank": r + 1}
            for r, (iid, s) in enumerate(zip(item_ids, scores))
        ]

    def batch_recommend(
        self,
        user_ids_raw: list,
        top_k: int = 500,
    ) -> dict[str, list[dict]]:
        """Memory-aware batch recommendation."""
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb < self.memory_threshold_gb:
            print(f"[Predictor] Low RAM ({avail_gb:.1f}GB) — reducing batch top_k to 100")
            top_k = 100

        return {uid: self.recommend(uid, top_k=top_k) for uid in user_ids_raw}
