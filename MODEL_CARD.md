# Model Card — Fraud Detection Ensemble

A short, honest description of what this model is, how it was measured, and
where it should *not* be trusted. Metrics are reproducible with
`python scripts/eval.py`.

## Model details

- **Type:** Weighted soft-voting ensemble — `0.55 * XGBoost + 0.45 * LightGBM`,
  each wrapped in isotonic probability calibration (`CalibratedClassifierCV`,
  prefit on a held-out calibration split).
- **Inputs:** 33 features — the 30 raw columns (`V1..V28` anonymized PCA
  components, `Time`, `Amount`) plus three engineered features
  (`hour_of_day`, `log_amount`, `amount_band`).
- **Output:** a calibrated probability of fraud in `[0, 1]`.
- **Decision policy (defaults in `score_api.py`):** approve `< 0.43`,
  manual review `[0.43, 0.95)`, auto-block `>= 0.95`. The F1-optimal point on
  the PR curve is `t = 0.55`.
- **Version:** `v1.2.0`. **Artifact:** `artifacts/model.pkl` (joblib).
  Also exported to ONNX (`docs/model/`) for client-side inference.

## Training data

- **Source:** ULB Credit Card Fraud Detection dataset (Kaggle / OpenML id 1597)
  — real card transactions by European cardholders over two days in September
  2013.
- **Size / imbalance:** 284,807 transactions, 492 fraud (0.172% positive rate).
- **Splits:** 80/20 stratified train/test (`random_state=42`); the train half
  is further split 85/15 to create a calibration set the model never fits on.
- **Imbalance handling:** SMOTE oversampling to a 0.10 minority ratio on the
  training fold, plus class weighting in both boosters.

## Metrics (held-out 20% test split: 56,962 transactions, 98 fraud)

| Metric | Value |
|---|---|
| ROC-AUC | 0.984 |
| PR-AUC (average precision) | 0.877 |
| Recall @ 0.1% false-positive rate | 0.898 |
| F1 @ t=0.55 | 0.857 |

PR-AUC is the headline metric: at this imbalance, ROC-AUC is optimistic and
accuracy is meaningless (a constant "never fraud" predictor scores 99.83%).

## Intended use

- **Intended:** a portfolio/educational demonstration of an imbalanced-
  classification pipeline and its operational surface (calibration, threshold
  policy, prediction logging, drift replay, client-side serving).
- **Not intended:** production fraud decisions. It has not been validated on
  any real payment stream, is trained on a single 2013 European dataset, and
  carries no fairness, adversarial-robustness, or regulatory review.

## Limitations and caveats

- **It misses fraud.** Recall at the strict 0.1% FPR operating point is 0.898 —
  roughly 1 in 10 frauds is not caught. The demo deliberately surfaces one such
  miss (the "Hardest fraud" preset).
- **Anonymized features.** `V1..V28` are PCA components, so per-feature
  explanations (the occlusion attributions in the demo) are not human-
  interpretable as "merchant category" or "device" — they are directional only.
- **Drift.** Fraud patterns move; a 2013 snapshot will not reflect current
  tactics. The pipeline includes prediction logging and replay hooks precisely
  so a deployed version could be retrained, but no live retraining is wired up.
- **Calibration is sparse in the mid-range.** With only 98 test frauds, the
  0.3–0.7 probability bins hold very few cases, so the reliability curve is
  noisy there even though it is exact at the extremes.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/download_data.py        # Kaggle, or OpenML fallback (no creds)
python scripts/train.py                # ~4 min
python scripts/eval.py                 # prints the metrics above
```
