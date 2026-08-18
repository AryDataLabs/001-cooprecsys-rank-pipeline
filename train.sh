#!/usr/bin/env bash
# train.sh — End-to-end training pipeline for cooprecsys AryColBring
set -euo pipefail

MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-cooprecsys}"
N_FACTORS="${N_FACTORS:-128}"
N_ITER="${N_ITER:-30}"
ALPHA="${ALPHA:-40.0}"
REG="${REG:-0.01}"

echo "🚀 [train.sh] Starting cooprecsys training pipeline"
echo "   Experiment : $MLFLOW_EXPERIMENT"
echo "   n_factors  : $N_FACTORS"

# 1. Build pseudo-ratings
python - << PYEOF
import pandas as pd
from cooprecsys.preprocessing.counterfeit_core import CounterFeitCore
from cooprecsys.preprocessing.cf_preprocess import build_interaction_matrix, temporal_train_test_split
import pickle, os

print("📦 Loading raw interactions...")
df = pd.read_parquet("data/raw/interactions.parquet")

print("⚙️  Building pseudo-ratings...")
engine = CounterFeitCore(method="weighted")
rated = engine.build_pseudo_ratings(df)

train_df, test_df = temporal_train_test_split(rated, timestamp_col="timestamp")
train_mat, user_enc, item_enc = build_interaction_matrix(train_df)
test_mat, _, _  = build_interaction_matrix(test_df)

os.makedirs("data/processed", exist_ok=True)
import scipy.sparse as sp
sp.save_npz("data/processed/train_matrix.npz", train_mat)
sp.save_npz("data/processed/test_matrix.npz",  test_mat)
with open("data/processed/encoders.pkl", "wb") as f:
    pickle.dump({"user_encoder": user_enc, "item_encoder": item_enc}, f)
print("✅ Matrices saved")
PYEOF

# 2. Train ALS model with MLflow tracking
python - << PYEOF
import scipy.sparse as sp, pickle, mlflow
from cooprecsys.models.trainer import AryColBringTrainer

train_mat = sp.load_npz("data/processed/train_matrix.npz")
test_mat  = sp.load_npz("data/processed/test_matrix.npz")

mlflow.set_experiment("$MLFLOW_EXPERIMENT")
with mlflow.start_run(run_name="AryColBring_v_$(date +%Y%m%d_%H%M)"):
    trainer = AryColBringTrainer(
        n_factors=$N_FACTORS, n_iterations=$N_ITER,
        regularization=$REG, alpha=$ALPHA
    )
    trainer.fit(train_mat, show_progress=True)
    metrics = trainer.evaluate(test_mat, k_values=[5, 10, 20])

    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlflow.log_params({"n_factors": $N_FACTORS, "n_iterations": $N_ITER})
    mlflow.sklearn.log_model(trainer, "AryColBring_model",
                             registered_model_name="AryColBring")

    print("📊 Metrics:", metrics)
print("✅ Training complete")
PYEOF

echo "🎉 [train.sh] Pipeline finished"
