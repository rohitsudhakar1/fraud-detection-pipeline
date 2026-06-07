"""Asynchronous prediction logger to PostgreSQL.

Every /score request writes one row so we can replay traffic later for
offline analysis, drift dashboards, or shadow-mode evaluation when a new
model version is trained.

Fire-and-forget: writes happen on a background thread so the scoring path
stays under 50ms p50. If the DB is down, we drop the log line rather than
fail the scoring request.

Schema (idempotent):

    CREATE TABLE IF NOT EXISTS predictions (
        id              BIGSERIAL PRIMARY KEY,
        ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        model_version   TEXT NOT NULL,
        feature_hash    TEXT NOT NULL,
        score           DOUBLE PRECISION NOT NULL,
        decision        TEXT NOT NULL,
        latency_ms      INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions (ts DESC);
    CREATE INDEX IF NOT EXISTS idx_predictions_hash ON predictions (feature_hash);
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)

_DSN = os.environ.get("PREDICTION_LOG_DSN")  # e.g. postgres://user:pw@host:5432/db
_QUEUE: "queue.Queue[tuple]" = queue.Queue(maxsize=10_000)
_WORKER: Optional[threading.Thread] = None
_STOP = threading.Event()


def _connect():
    import psycopg2

    return psycopg2.connect(_DSN)


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_version TEXT NOT NULL,
                feature_hash TEXT NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                decision TEXT NOT NULL,
                latency_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions (ts DESC);
            CREATE INDEX IF NOT EXISTS idx_predictions_hash ON predictions (feature_hash);
        """)
    conn.commit()


def _drain_loop() -> None:
    if not _DSN:
        return
    try:
        conn = _connect()
        _ensure_schema(conn)
    except Exception as e:
        log.warning("prediction-log connect failed (%s) - logger disabled", e)
        return

    while not _STOP.is_set():
        try:
            item = _QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (model_version, feature_hash, score, decision, latency_ms)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    item,
                )
            conn.commit()
        except Exception as e:
            log.warning("prediction-log insert failed: %s", e)


def start_logger() -> None:
    """Spin up the background writer. Idempotent."""
    global _WORKER
    if _WORKER and _WORKER.is_alive():
        return
    _WORKER = threading.Thread(target=_drain_loop, name="prediction-logger", daemon=True)
    _WORKER.start()


def log_prediction(
    *, model_version: str, feature_hash: str, score: float, decision: str, latency_ms: int
) -> None:
    """Enqueue a prediction for async write. Drops silently if queue is full."""
    if not _DSN:
        return
    try:
        _QUEUE.put_nowait((model_version, feature_hash, score, decision, latency_ms))
    except queue.Full:
        log.debug("prediction-log queue full, dropping")


@contextmanager
def shutdown_on_exit():
    """Context manager for graceful shutdown (used in tests)."""
    try:
        yield
    finally:
        _STOP.set()
