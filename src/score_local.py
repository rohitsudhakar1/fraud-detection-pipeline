"""Tiny CLI for scoring a single transaction from the command line.

    python -m src.score_local --model artifacts/model.pkl --json '{"Time":..., "V1":..., ...}'

Useful for smoke-testing a freshly trained bundle without spinning up FastAPI.
"""
from __future__ import annotations

import argparse
import json

from .model import load_bundle, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/model.pkl")
    parser.add_argument("--json", required=True, help="JSON object with V1..V28, Time, Amount")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    row = json.loads(args.json)
    p = score(bundle, row)
    print(json.dumps({"score": round(p, 6), "model_version": bundle["model_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
