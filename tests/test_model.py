"""Smoke test for the training pipeline on a tiny synthetic sample.

We don't run a full Kaggle-scale training here — that's gated behind real
data. Instead we generate a tiny imbalanced dataset and confirm the
training function returns a usable bundle and metrics dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synth_dataset(n_legit: int = 2_000, n_fraud: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(2.5, 1, size=(n_fraud, 28))   # shifted distribution → learnable
    X = np.vstack([legit, fraud])
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = rng.integers(0, 172_800, size=len(df))
    df["Amount"] = rng.exponential(50, size=len(df))
    df["Class"] = [0] * n_legit + [1] * n_fraud
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.mark.slow
def test_train_ensemble_returns_bundle_and_metrics():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    pytest.importorskip("imblearn")
    from src.model import train_ensemble

    df = _synth_dataset()
    bundle, metrics = train_ensemble(df)

    # Bundle shape
    assert {"xgb", "lgb", "weight_xgb", "feature_cols", "model_version"} <= set(bundle.keys())

    # Metrics shape + plausibility
    for key in ("roc_auc", "pr_auc", "best_threshold", "best_f1", "recall_at_0_1pct_fpr"):
        assert key in metrics
    # Synthetic data is separable, so we expect strong ROC-AUC.
    assert metrics["roc_auc"] > 0.95
