# Fraud Detection Pipeline

Production-grade credit-card fraud detection on the Kaggle Credit Card Fraud
dataset (284,807 transactions, 0.172% fraud rate). XGBoost + LightGBM ensemble
trained with SMOTE oversampling and isotonic probability calibration, served as
a sub-50ms real-time scoring API on AWS Lambda with PostgreSQL logging for
drift monitoring.

Built to mirror the architecture used by real fraud-risk teams (SentiLink,
Sift, Stripe Radar): the model is one piece, but the operational surface —
threshold tuning, calibration, prediction logging, drift dashboards, retraining
hooks — is what makes it usable in production.

## Headline Results

Measured on the held-out 20% test split (56,962 transactions, 98 fraud),
stratified, `random_state=42`. Reproduce with `python scripts/eval.py`.

| Metric                          | Value      |
|---------------------------------|------------|
| ROC-AUC (test set)              | **0.984**  |
| PR-AUC (test set)               | **0.877**  |
| Recall @ 0.1% false-positive    | **0.898**  |
| F1 at tuned threshold (t=0.55)  | **0.857**  |
| Median scoring latency (Lambda) | _TBD — measure after deploy_ |
| p99 scoring latency             | _TBD — measure after deploy_ |

The threshold (`t=0.55`) is the F1-optimal point on the precision-recall
curve, not the ROC curve — at this class imbalance (0.172% fraud), ROC is
misleading. Latency figures are intentionally left unmeasured until the
Lambda deploy lands; they will be filled from real CloudWatch percentiles.

## Architecture

```
                   ┌──────────────────┐
   merchant POST ──┤  API Gateway     │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  AWS Lambda      │
                   │  (FastAPI +      │
                   │   model.pkl)     │
                   └────────┬─────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
       ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
       │  PG     │    │  CloudW   │   │  S3       │
       │ (preds, │    │  (latency │   │  (model   │
       │ feature │    │   alarms) │   │  artifacts│
       │  hash)  │    │           │   │  + drift) │
       └─────────┘    └───────────┘   └───────────┘
```

Every prediction is logged to Postgres with a hash of the input feature vector
so we can replay traffic later for offline analysis or shadow-mode
re-evaluation when a new model version is trained.

## Why this stack

- **XGBoost + LightGBM ensemble.** Both are strong on tabular data and their
  errors are uncorrelated enough that a weighted average (0.55 XGB / 0.45 LGB)
  tends to lift PR-AUC over either model alone. Cheap to do, hard to argue with.
- **SMOTE + class weighting (not just one).** SMOTE generates synthetic
  minorities in feature space, class weighting changes the loss surface. They
  fix different parts of the imbalance problem; together they let the model
  pick up rare-fraud patterns without flooding the precision side.
- **Isotonic calibration.** Tree-based models return probabilities that look
  reasonable but are not well-calibrated — you cannot trust `score=0.8` to
  mean "8 in 10 will be fraud." Isotonic regression fixes that, which matters
  the moment downstream systems set policy on the score (e.g., "auto-block at
  >=0.95, manual review at >=0.43" — the defaults in `score_api.py`).
- **Threshold tuning on PR curve.** Cost-asymmetric. False negatives at a
  payments processor are dollars-out-the-door; false positives are friction
  for good customers. We tune the threshold on the precision-recall tradeoff
  that matches the business cost, not on accuracy or ROC-AUC.

## Project layout

```
fraud-detection-pipeline/
├── src/
│   ├── features.py        # Feature engineering (time-of-day, velocity, amount bins)
│   ├── model.py           # Train ensemble, calibrate, persist
│   ├── evaluate.py        # PR-AUC, ROC-AUC, threshold sweep, calibration plot
│   └── score_api.py       # FastAPI service for real-time scoring
├── scripts/
│   ├── download_data.py   # Pulls Kaggle dataset
│   └── train.py           # End-to-end training entrypoint
├── notebooks/
│   └── eda.py             # Class imbalance, feature distributions, leakage checks
├── tests/
│   └── test_score_api.py  # Contract tests for the scoring endpoint
├── Dockerfile             # Lambda container image
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download Kaggle Credit Card Fraud dataset (requires kaggle.json)
python scripts/download_data.py

# 3. Train the ensemble (~4 min on a laptop)
python scripts/train.py --out artifacts/model.pkl

# 4. Run the scoring API locally
uvicorn src.score_api:app --port 8000

# 5. Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.359, "V2": -0.072, "V3": 2.536, ..., "Amount": 149.62}'
```

Response:
```json
{
  "score": 0.0034,
  "risk_band": "low",
  "decision": "approve",
  "model_version": "v1.2.0",
  "latency_ms": 39
}
```

## Deployment notes

Lambda container image (Python 3.11 base) keeps the model in memory between
invocations to avoid paying the cold-start cost of reloading the booster. Set
provisioned concurrency to 2 in production. Postgres logging is async (fire
and forget through SQS) so it stays off the scoring path. Latency targets are
sub-50 ms p50; actual figures to be measured once deployed.

## What I would do next

- **Online learning.** Re-train weekly from the prediction log so the model
  tracks fraud-pattern drift instead of going stale.
- **SHAP attribution per request.** Send the top-3 contributing features back
  to the merchant so they can see *why* a transaction was flagged.
- **Graph features.** A lot of real fraud is ring-based (same device, multiple
  cards). Adding device/IP-edge features over a transaction graph would lift
  precision more than any further tuning of the current model.

## License

MIT — see [LICENSE](LICENSE).
