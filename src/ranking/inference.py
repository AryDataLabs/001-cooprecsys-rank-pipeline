"""
inference.py — Batch and real-time recommendation inference engine.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cooprecsys.models.predictor import AryColBringPredictor


class BatchInferenceEngine:
    """
    Generates candidate pools for all active users in batches.

    Parameters
    ----------
    predictor   : AryColBringPredictor
    top_k       : int   number of candidates per user
    n_workers   : int   parallel workers (future: multiprocessing)
    output_path : str   directory to write Parquet candidate pool files
    tqdm_colour : str
    """

    def __init__(
        self,
        predictor: AryColBringPredictor,
        top_k: int = 500,
        n_workers: int = 4,
        output_path: str = "data/processed/candidate_pools/",
        tqdm_colour: str = "#05ad46",
    ):
        self.predictor = predictor
        self.top_k = top_k
        self.n_workers = n_workers
        self.output_path = Path(output_path)
        self.tqdm_colour = tqdm_colour
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run_batch(self, user_ids: list, chunk_size: int = 1000) -> None:
        """
        Run batch inference and write Parquet files per chunk.

        Parameters
        ----------
        user_ids   : list of raw user IDs
        chunk_size : int
        """
        chunks = [user_ids[i:i + chunk_size] for i in range(0, len(user_ids), chunk_size)]
        print(f"[BatchInference] {len(user_ids)} users | {len(chunks)} chunks | top_k={self.top_k}")

        for chunk_idx, chunk in enumerate(
            tqdm(chunks, desc="Batch inference", colour=self.tqdm_colour)
        ):
            records = []
            for uid in chunk:
                recs = self.predictor.recommend(uid, top_k=self.top_k)
                for r in recs:
                    records.append({"user_id": uid, **r})

            if records:
                df = pd.DataFrame(records)
                out_file = self.output_path / f"candidates_chunk_{chunk_idx:04d}.parquet"
                df.to_parquet(out_file, index=False)

        print(f"[BatchInference] ✅ Written to {self.output_path}")

    def load_all_candidates(self) -> pd.DataFrame:
        """Load all written candidate pool Parquet files into one DataFrame."""
        files = sorted(self.output_path.glob("candidates_chunk_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No candidate files found in {self.output_path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
