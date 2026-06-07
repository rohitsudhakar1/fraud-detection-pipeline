"""Train an XGBoost + LightGBM ensemble with SMOTE oversampling and
isotonic probability calibration.

Why this shape:

- The fraud rate is 0.172%. Plain training under-weights the minority class
  so much that the model collapses to the all-zero baseline. SMOTE generates
  synthetic minority samples in feature space; class weighting in the
  boosters changes the loss surface. The two together work better than
  either alone (we tried both ablations).

- Trees output uncalibrated probabilities. We wrap the trained ensemble in
  isotonic regression on a held-out calibration split so downstream systems
  can set policy on the score (e.g., auto-block at >0.95).

- The threshold is tuned on the precision-recall curve, not on accuracy or
  ROC. At this class imbalance, ROC is misleading.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .features import build_features, feature_columns

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    learning_rate=0.05,
    max_depth=6,
    n_estimators=400,
    min_child_weight=4,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=200,
    tree_method="hist",
    random_state=42,
)

LGB_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    n_estimators=400,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    is_unbalance=True,
    random_state=42,
)


def train_ensemble(df: pd.DataFrame, *, ensemble_weight_xgb: float = 0.55):
    """Train the ensemble on `df`, return (model_bundle, holdout_metrics).

    Heavy ML deps are imported lazily so the scoring service (which only
    needs the persisted bundle + sklearn) can boot without xgboost /
    lightgbm / imblearn installed on the runtime image.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    df = build_features(df)
    X = df[feature_columns()]
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    smote = SMOTE(sampling_strategy=0.10, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_clf.fit(X_train_smote, y_train_smote)

    lgb_clf = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_clf.fit(X_train_smote, y_train_smote)

    # Calibrate each base learner independently on the held-out calibration
    # split — isotonic regression on tree probabilities.
    xgb_cal = CalibratedClassifierCV(xgb_clf, method="isotonic", cv="prefit")
    xgb_cal.fit(X_calib, y_calib)

    lgb_cal = CalibratedClassifierCV(lgb_clf, method="isotonic", cv="prefit")
    lgb_cal.fit(X_calib, y_calib)

    def predict_proba(X):
        p_xgb = xgb_cal.predict_proba(X)[:, 1]
        p_lgb = lgb_cal.predict_proba(X)[:, 1]
        return ensemble_weight_xgb * p_xgb + (1 - ensemble_weight_xgb) * p_lgb

    bundle = dict(
        xgb=xgb_cal,
        lgb=lgb_cal,
        weight_xgb=ensemble_weight_xgb,
        feature_cols=feature_columns(),
        model_version="v1.2.0",
    )

    from .evaluate import score_holdout
    metrics = score_holdout(predict_proba, X_test, y_test)

    return bundle, metrics


def save_bundle(bundle: dict, path: str) -> None:
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def score(bundle: dict, row: dict) -> float:
    """Score a single transaction. Used by the API."""
    df = pd.DataFrame([row])
    df = build_features(df)
    X = df[bundle["feature_cols"]]
    p_xgb = bundle["xgb"].predict_proba(X)[:, 1]
    p_lgb = bundle["lgb"].predict_proba(X)[:, 1]
    w = bundle["weight_xgb"]
    return float(w * p_xgb[0] + (1 - w) * p_lgb[0])
