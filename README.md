# cooprecsys

> **Production-grade Collaborative Filtering Recommendation System**
> From implicit behavioral signals to ALS-WR embeddings with Cython-accelerated inference.

[![CI](https://github.com/masterofray/cooprecsys/actions/workflows/ci.yml/badge.svg)](https://github.com/masterofray/cooprecsys/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/mlflow-tracked-orange)](https://mlflow.org/)

## Architecture
```
Raw Interactions → CounterFeitCore (pseudo-rating) → ALS Training → AryColBringPredictor → FastAPI
```

## Quick Start
```bash
# 1. Install
pip install -r requirements.txt

# 2. Build Cython extension
bash scripts/build_cython.sh

# 3. Train
bash scripts/train.sh

# 4. Serve
bash scripts/serve.sh
```

## Key Components
| Module | Description |
|---|---|
| `preprocessing/counterfeit_core.py` | Pseudo-rating engine (5 methods, DuckDB CTEs) |
| `preprocessing/cf_preprocess.py` | Sparse matrix builder |
| `_cy/_cy_predict.pyx` | Cython-accelerated dot-product scoring |
| `models/trainer.py` | AryColBringTrainer (ALS-WR + MLflow) |
| `models/predictor.py` | Production inference with memory-aware batching |
| `ranking/inference.py` | Batch candidate generation engine |
| `api/serve.py` | FastAPI recommendation endpoint |

## Notebooks
- `notebooks/01_collaborative_filtering_tutorial.ipynb` — Full walkthrough

## Citation
```
Aryanto (masterofray). cooprecsys. 2024. https://github.com/masterofray/cooprecsys
```
