"""Export real diagnostic curves for the demo's 'Model diagnostics' charts.

Computes, on the deterministic held-out test split (same as training):
  - precision / recall / F1 across a fine threshold grid
  - a calibration reliability curve (predicted vs observed fraud rate)
  - the precision-recall curve
  - an illustrative decision-cost curve (stated FN:FP cost ratio)

Writes docs/model/curves.json. Everything here is measured, not assumed,
except the cost ratio which is labelled as illustrative in the UI.

    python scripts/export_curves.py --data data/creditcard.csv --model artifacts/model.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.features import build_features, feature_columns  # noqa: E402
from src.model import load_bundle  # noqa: E402

FN_FP_RATIO = 100  # illustrative: a missed fraud costs ~100x a false-positive's friction


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--model", default="artifacts/model.pkl")
    args = ap.parse_args()

    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split

    df = build_features(pd.read_csv(args.data))
    X, y = df[feature_columns()], df["Class"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    b = load_bundle(args.model)
    w = b["weight_xgb"]
    proba = w * b["xgb"].predict_proba(X_test)[:, 1] + (1 - w) * b["lgb"].predict_proba(X_test)[:, 1]
    yv = y_test.values
    P, N = int(yv.sum()), int((yv == 0).sum())

    # --- threshold sweep ---
    sweep = []
    for t in np.linspace(0.0, 1.0, 101):
        pred = proba >= t
        tp = int((pred & (yv == 1)).sum()); fp = int((pred & (yv == 0)).sum())
        fn = int((~pred & (yv == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        cost = FN_FP_RATIO * fn + fp
        sweep.append({"t": round(float(t), 2), "precision": round(prec, 4),
                      "recall": round(rec, 4), "f1": round(f1, 4), "cost": cost})

    # --- calibration reliability (10 equal-width bins) ---
    calib = []
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        m = (proba >= lo) & (proba < hi if hi < 1.0 else proba <= 1.0)
        if m.sum() == 0:
            continue
        calib.append({"mid": round(float(lo + 0.05), 3),
                      "pred": round(float(proba[m].mean()), 4),
                      "obs": round(float(yv[m].mean()), 4),
                      "n": int(m.sum())})

    # --- PR curve, resampled on an even recall grid ---
    # precision_recall_curve piles thousands of points in the low-precision
    # tail (56k near-zero-prob legit rows); sampling by array index misses the
    # curve's shape. Resample by recall using interpolated precision (max
    # precision achievable at >= each recall), the standard PR-curve envelope.
    prec, rec, _ = precision_recall_curve(yv, proba)
    pr = []
    for rg in np.linspace(0, 1, 50):
        mask = rec >= rg
        p = float(prec[mask].max()) if mask.any() else float(prec[-1])
        pr.append({"r": round(float(rg), 4), "p": round(p, 4)})

    out = {
        "n_test": int(len(yv)), "n_fraud": P, "n_legit": N,
        "fn_fp_ratio": FN_FP_RATIO,
        "sweep": sweep, "calibration": calib, "pr": pr,
        "thresholds": {"review": 0.43, "block": 0.95, "f1opt": 0.55},
    }
    os.makedirs("docs/model", exist_ok=True)
    with open("docs/model/curves.json", "w") as f:
        json.dump(out, f)
    print(f"[curves] test={len(yv)} fraud={P}: wrote sweep({len(sweep)}), "
          f"calibration({len(calib)} bins), pr({len(pr)}) -> docs/model/curves.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
