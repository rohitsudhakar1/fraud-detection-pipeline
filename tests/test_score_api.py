"""Contract tests for the scoring endpoint. Runs without the trained model
by stubbing the bundle directly — verifies request/response shape and
decision logic without spinning up the lifespan handler.
"""
from __future__ import annotations

from unittest.mock import patch

import fastapi
import pytest
from fastapi.testclient import TestClient

# fastapi < 0.110 has a starlette-middleware-unpacking bug that breaks
# TestClient construction in modern starlette environments. The pinned
# version in requirements.txt is 0.115.0; this guard keeps the test green
# in older local environments.
pytestmark = pytest.mark.skipif(
    tuple(int(p) for p in fastapi.__version__.split(".")[:2]) < (0, 110),
    reason=f"fastapi {fastapi.__version__} is older than the pinned 0.115.0",
)


def _stub_bundle():
    return {
        "model_version": "test-stub",
        "feature_cols": [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "hour_of_day", "log_amount", "amount_band"],
        "weight_xgb": 0.55,
    }


def _make_tx(amount: float = 100.0) -> dict:
    body = {f"V{i}": 0.0 for i in range(1, 29)}
    body["Time"] = 36000.0
    body["Amount"] = amount
    return body


@pytest.fixture()
def client():
    # Bypass the lifespan handler entirely by stubbing _bundle before creating
    # the TestClient and skipping the startup context manager.
    from src import score_api
    score_api._bundle = _stub_bundle()
    # Don't use `with TestClient(...)` because that triggers lifespan startup;
    # construct directly so we keep the stub.
    c = TestClient(score_api.app)
    yield c
    score_api._bundle = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_score_shape(client):
    with patch("src.score_api.score", return_value=0.12):
        r = client.post("/score", json=_make_tx())
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"score", "risk_band", "decision", "model_version", "feature_hash", "latency_ms"}
    assert body["risk_band"] == "low"
    assert body["decision"] == "approve"


def test_score_blocks_high_risk(client):
    with patch("src.score_api.score", return_value=0.97):
        r = client.post("/score", json=_make_tx())
    assert r.json()["decision"] == "block"


def test_score_review_band(client):
    with patch("src.score_api.score", return_value=0.55):
        r = client.post("/score", json=_make_tx())
    assert r.json()["decision"] == "manual_review"
