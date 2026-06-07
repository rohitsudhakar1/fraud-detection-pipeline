"""Feature engineering unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_features, feature_columns, feature_hash


def test_build_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            **{f"V{i}": [0.1 * i] for i in range(1, 29)},
            "Time": [3600 * 14.5],
            "Amount": [149.62],
            "Class": [0],
        }
    )
    out = build_features(df)
    assert "hour_of_day" in out.columns
    assert "log_amount" in out.columns
    assert "amount_band" in out.columns
    assert out["hour_of_day"].iloc[0] == 14
    assert np.isclose(out["log_amount"].iloc[0], np.log1p(149.62))


def test_amount_band_is_monotone_in_amount():
    df = pd.DataFrame(
        {
            **{f"V{i}": [0.0] * 4 for i in range(1, 29)},
            "Time": [0.0, 0.0, 0.0, 0.0],
            "Amount": [0.5, 5.0, 100.0, 10_000.0],
        }
    )
    bands = build_features(df)["amount_band"].tolist()
    assert bands == sorted(bands)


def test_feature_hash_is_stable_under_reorder():
    a = {"V1": 1.234, "V2": 2.345, "Amount": 99.99}
    b = {"V2": 2.345, "Amount": 99.99, "V1": 1.234}
    assert feature_hash(a) == feature_hash(b)


def test_feature_columns_has_expected_shape():
    cols = feature_columns()
    # 28 PCA features + Time + Amount + 3 engineered (hour_of_day, log_amount, amount_band)
    assert len(cols) == 33
    assert cols[:28] == [f"V{i}" for i in range(1, 29)]
    assert {"hour_of_day", "log_amount", "amount_band"} <= set(cols)
