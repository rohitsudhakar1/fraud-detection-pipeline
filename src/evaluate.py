"""Evaluation: PR-AUC, ROC-AUC, threshold sweep, calibration plot.

Reported numbers in the README come from this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def score_holdout(predict_proba, X_test, y_test) -> dict:
    """Return a dict of headline metrics for the trained ensemble."""
    proba = predict_proba(X_test)
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    # Threshold tuned for max F1 on the PR curve.
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1 = (2 * precision * recall) / np.where(
        precision + recall == 0, 1, precision + recall
    )
    best_idx = int(np.argmax(f1[:-1]))
    best_t = float(thresholds[best_idx])

    # Recall at 0.1% false-positive rate (operating point a real fraud team
    # would care about — keeps friction under control).
    fpr_target = 0.001
    n_negative = int((y_test == 0).sum())
    max_fp_allowed = int(fpr_target * n_negative)

    sort_idx = np.argsort(-proba)
    y_sorted = y_test.values[sort_idx]
    proba_sorted = proba[sort_idx]
    cum_fp = np.cumsum(y_sorted == 0)
    cutoff = np.searchsorted(cum_fp, max_fp_allowed, side="right")
    recall_at_low_fpr = float((y_sorted[:cutoff] == 1).sum() / max((y_test == 1).sum(), 1))

    return dict(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        best_threshold=best_t,
        best_f1=float(f1[best_idx]),
        recall_at_0_1pct_fpr=recall_at_low_fpr,
    )


def threshold_sweep(predict_proba, X_test, y_test) -> pd.DataFrame:
    """For dashboards: precision/recall/F1 over a sweep of thresholds."""
    proba = predict_proba(X_test)
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        rows.append(dict(threshold=round(t, 3), precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn))
    return pd.DataFrame(rows)
