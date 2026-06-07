"""Download the Kaggle Credit Card Fraud dataset.

Requires the `kaggle` CLI to be configured (~/.kaggle/kaggle.json or
KAGGLE_USERNAME + KAGGLE_KEY env vars).
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    os.makedirs("data", exist_ok=True)
    target = "data/creditcard.csv"
    if os.path.exists(target):
        print(f"[download] {target} already exists, skipping")
        return 0
    cmd = ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", "data", "--unzip"]
    print(f"[download] running: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
