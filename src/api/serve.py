"""
serve.py — FastAPI serving endpoint for cooprecsys AryColBring.
"""
import os, pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.sklearn

from .schemas import RecommendRequest, RecommendResponse, RecommendedItem
from cooprecsys.models.predictor import AryColBringPredictor

_predictor: AryColBringPredictor | None = None
MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/AryColBring/Production")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

INTERACTION_THRESHOLD_COLD = int(os.getenv("COLD_START_THRESHOLD", "5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    print(f"[serve] Loading model from {MODEL_URI} ...")
    try:
        raw = mlflow.sklearn.load_model(MODEL_URI)
        with open(os.getenv("ENCODER_PATH", "encoders.pkl"), "rb") as f:
            encs = pickle.load(f)
        _predictor = AryColBringPredictor(
            model=raw,
            user_encoder=encs["user_encoder"],
            item_encoder=encs["item_encoder"],
        )
        print("[serve] ✅ Model loaded")
    except Exception as e:
        print(f"[serve] ⚠ Could not load model: {e}")
    yield


app = FastAPI(title="cooprecsys AryColBring API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_VERSION}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    recs = _predictor.recommend(req.user_id, top_k=req.top_k)
    if not recs:
        raise HTTPException(status_code=404, detail="User not found or no recommendations")
    return RecommendResponse(
        user_id=req.user_id,
        recommendations=[RecommendedItem(**r) for r in recs],
        model_version=MODEL_VERSION,
        strategy="arycolbring_cf",
    )
