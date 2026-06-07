"""End-to-end training entrypoint.

    python scripts/train.py --data data/creditcard.csv --out artifacts/model.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# Windows consoles default to cp1252; force UTF-8 so log glyphs don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import save_bundle, train_ensemble  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--out", default="artifacts/model.pkl")
    args = parser.parse_args()

    print(f"[train] loading {args.data}")
    df = pd.read_csv(args.data)
    print(f"[train] rows={len(df):,} fraud={int(df['Class'].sum())} ({df['Class'].mean()*100:.3f}%)")

    print("[train] fitting XGBoost + LightGBM ensemble with SMOTE + isotonic calibration")
    bundle, metrics = train_ensemble(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_bundle(bundle, args.out)
    print(f"[train] saved bundle → {args.out}")
    print("[train] holdout metrics:")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
