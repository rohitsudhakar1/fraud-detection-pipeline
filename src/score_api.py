"""Real-time scoring API.

Lambda + API Gateway in production, uvicorn locally. The model bundle is
loaded once at module import so warm Lambda invocations skip the 1.2s
booster reload.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import db
from .features import feature_hash
from .model import load_bundle, score

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pkl")
DECISION_AUTO_BLOCK = float(os.environ.get("DECISION_AUTO_BLOCK", "0.95"))
DECISION_REVIEW = float(os.environ.get("DECISION_REVIEW", "0.43"))

_bundle: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    _bundle = load_bundle(MODEL_PATH)
    db.start_logger()
    yield


app = FastAPI(title="Fraud Detection Scoring API", version="1.2.0", lifespan=lifespan)


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since first transaction in dataset window")
    Amount: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float


class ScoreResponse(BaseModel):
    score: float
    risk_band: str
    decision: str
    model_version: str
    feature_hash: str
    latency_ms: int


def _decision(p: float) -> tuple[str, str]:
    if p >= DECISION_AUTO_BLOCK:
        return "high", "block"
    if p >= DECISION_REVIEW:
        return "medium", "manual_review"
    return "low", "approve"


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _bundle is not None}


@app.post("/score", response_model=ScoreResponse)
def score_transaction(tx: Transaction):
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    t0 = time.perf_counter()
    row = tx.model_dump()
    p = score(_bundle, row)
    risk_band, decision = _decision(p)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    fh = feature_hash(row)
    db.log_prediction(
        model_version=_bundle["model_version"],
        feature_hash=fh,
        score=p,
        decision=decision,
        latency_ms=latency_ms,
    )
    return ScoreResponse(
        score=round(p, 4),
        risk_band=risk_band,
        decision=decision,
        model_version=_bundle["model_version"],
        feature_hash=fh,
        latency_ms=latency_ms,
    )
