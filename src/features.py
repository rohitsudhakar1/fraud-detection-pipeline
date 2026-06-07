"""Feature engineering for the credit-card fraud dataset.

The raw Kaggle dataset gives us 28 PCA components (V1..V28) plus Time and
Amount. The PCA features are already anonymized and decorrelated, so most
classical feature engineering does not apply. The two raw columns are where
the value is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the raw dataframe with engineered features.

    - Hour-of-day from the Time column (Time is seconds-since-first-tx in
      the Kaggle dump; the dataset spans two days).
    - Log-amount, since the amount distribution is heavy-tailed.
    - Amount-band ordinal so the model has a coarse view of magnitude.
    """
    out = df.copy()
    out["hour_of_day"] = (out["Time"] // 3600 % 24).astype(int)
    out["log_amount"] = np.log1p(out["Amount"])
    out["amount_band"] = pd.cut(
        out["Amount"],
        bins=[-0.01, 1, 10, 50, 200, 1_000, 5_000, np.inf],
        labels=False,
    ).astype(int)
    return out


def feature_columns() -> list[str]:
    return RAW_FEATURE_COLS + ["hour_of_day", "log_amount", "amount_band"]


def feature_hash(row: dict) -> str:
    """Stable hash of a feature row for prediction-log replay.

    We hash the rounded feature vector so two near-identical requests collide,
    which makes shadow-mode A/B replay cheaper to bucket.
    """
    import hashlib

    parts = [f"{k}={round(float(row[k]), 4)}" for k in sorted(row)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
