"""Download the Kaggle Credit Card Fraud dataset.

Primary path uses the `kaggle` CLI (requires ~/.kaggle/kaggle.json or
KAGGLE_USERNAME + KAGGLE_KEY env vars). If the CLI is unavailable or
unauthenticated, falls back to the same dataset mirrored on OpenML
(dataset id 1597), which needs no credentials. Either path produces an
identical `data/creditcard.csv` (284,807 rows, columns V1..V28, Time,
Amount, Class).
"""
from __future__ import annotations

import os
import subprocess
import sys

TARGET = "data/creditcard.csv"
OPENML_ID = 1597  # ULB credit-card fraud dataset on OpenML


def _via_kaggle() -> int:
    cmd = ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", "data", "--unzip"]
    print(f"[download] trying kaggle: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[download] kaggle CLI not found")
        return 1


def _via_openml() -> int:
    print(f"[download] falling back to OpenML (dataset id {OPENML_ID}, no credentials needed)")
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True)
    df = bunch.frame
    # OpenML names the target "Class" already; ensure it is integer 0/1.
    df["Class"] = df["Class"].astype(int)
    df.to_csv(TARGET, index=False)
    print(f"[download] wrote {TARGET} rows={len(df):,} fraud={int(df['Class'].sum())}")
    return 0


def main() -> int:
    os.makedirs("data", exist_ok=True)
    if os.path.exists(TARGET):
        print(f"[download] {TARGET} already exists, skipping")
        return 0
    if _via_kaggle() == 0 and os.path.exists(TARGET):
        return 0
    return _via_openml()


if __name__ == "__main__":
    sys.exit(main())
