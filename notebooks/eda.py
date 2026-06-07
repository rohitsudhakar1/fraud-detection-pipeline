"""Exploratory data analysis on the credit-card fraud dataset.

Run as a script (`python notebooks/eda.py`) or paste into a notebook cell.
Validates the assumptions we make in the modeling stage:

- Class imbalance is 0.172% — we will not use accuracy as the metric.
- Time + Amount carry signal beyond the PCA features.
- No row-level leakage (Time is monotonic, fraud is spread across the window).
"""
from __future__ import annotations

import pandas as pd


def main() -> None:
    df = pd.read_csv("data/creditcard.csv")
    print(f"rows = {len(df):,}")
    print(f"fraud rate = {df['Class'].mean()*100:.3f}%")
    print(f"time span = {df['Time'].max() / 3600:.1f} hours")

    print("\nfraud rate by hour-of-day:")
    df["hour"] = (df["Time"] // 3600 % 24).astype(int)
    by_hour = df.groupby("hour")["Class"].mean() * 100
    print(by_hour.round(3).to_string())

    print("\namount distribution (fraud vs legit):")
    print(df.groupby("Class")["Amount"].describe()[["mean", "50%", "max"]].round(2))


if __name__ == "__main__":
    main()
