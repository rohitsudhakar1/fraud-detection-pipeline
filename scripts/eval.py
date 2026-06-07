"""Evaluate a trained bundle on the deterministic held-out test split.

Reproduces the exact train/test split used in training (stratified,
random_state=42) and reports the headline metrics from the saved model
without retraining. Also emits a handful of real example transactions
(highest-scoring frauds, lowest-scoring legitimate) for demos.

    python scripts/eval.py --data data/creditcard.csv --model artifacts/model.pkl
    python scripts/eval.py --examples examples.json   # also dump demo examples
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

from src.evaluate import score_holdout  # noqa: E402
from src.features import build_features, feature_columns  # noqa: E402
from src.model import load_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--model", default="artifacts/model.pkl")
    parser.add_argument("--examples", default=None, help="optional path to dump demo examples JSON")
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    df = build_features(pd.read_csv(args.data))
    X = df[feature_columns()]
    y = df["Class"].astype(int)
    # Identical split to train_ensemble: 20% test, stratified, seed 42.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    bundle = load_bundle(args.model)
    w = bundle["weight_xgb"]

    def predict_proba(Xin):
        p_xgb = bundle["xgb"].predict_proba(Xin)[:, 1]
        p_lgb = bundle["lgb"].predict_proba(Xin)[:, 1]
        return w * p_xgb + (1 - w) * p_lgb

    metrics = score_holdout(predict_proba, X_test, y_test)
    print(json.dumps(metrics, indent=2))

    if args.examples:
        proba = predict_proba(X_test)
        raw_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        test_raw = df.loc[X_test.index, raw_cols]
        idx = X_test.index.values
        yv = y_test.values
        asc = np.argsort(proba)        # low -> high score
        desc = asc[::-1]
        fraud_pos = [i for i in asc if yv[i] == 1]
        legit_pos = [i for i in asc if yv[i] == 0]
        # Span the decision surface with real rows the model actually saw:
        picks = [
            ("Clear fraud", desc[[i for i in range(len(desc)) if yv[desc[i]] == 1][0]]),
            ("Hardest fraud (lowest-scored true fraud)", fraud_pos[0]),
            ("Most fraud-like legit (top-scored true legit)", legit_pos[-1]),
            ("Clear legit", legit_pos[0]),
        ]
        examples = []
        for note, i in picks:
            row = test_raw.loc[idx[i]].to_dict()
            examples.append({
                "label": "fraud" if yv[i] == 1 else "legit",
                "note": note,
                "actual_class": int(yv[i]),
                "score": round(float(proba[i]), 4),
                "tx": {k: round(float(v), 4) for k, v in row.items()},
            })
        with open(args.examples, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"[eval] wrote {len(examples)} demo examples -> {args.examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
