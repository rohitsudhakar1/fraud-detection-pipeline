-- Prediction log schema. The score_api.db module ensures this on startup,
-- but you may want to provision it explicitly with managed migrations.

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL,
    feature_hash    TEXT NOT NULL,
    score           DOUBLE PRECISION NOT NULL,
    decision        TEXT NOT NULL,
    latency_ms      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_ts   ON predictions (ts DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_hash ON predictions (feature_hash);

-- Drift-monitoring view: score distribution by hour.
-- Plug into Grafana / Streamlit to spot population shift between training
-- distribution and live traffic.
CREATE OR REPLACE VIEW score_distribution_hourly AS
SELECT
    date_trunc('hour', ts)                                AS bucket,
    model_version,
    count(*)                                              AS n,
    avg(score)                                            AS mean_score,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY score)   AS p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY score)   AS p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY score)   AS p99,
    sum(CASE WHEN decision = 'block'         THEN 1 ELSE 0 END) AS n_block,
    sum(CASE WHEN decision = 'manual_review' THEN 1 ELSE 0 END) AS n_review,
    sum(CASE WHEN decision = 'approve'       THEN 1 ELSE 0 END) AS n_approve
FROM predictions
GROUP BY 1, 2
ORDER BY 1 DESC;

-- Latency percentiles (used by the CloudWatch dashboard).
CREATE OR REPLACE VIEW latency_distribution_hourly AS
SELECT
    date_trunc('hour', ts)                                  AS bucket,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms,
    count(*)                                                AS n
FROM predictions
GROUP BY 1
ORDER BY 1 DESC;
