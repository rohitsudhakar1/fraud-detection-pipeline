"""Export the trained ensemble for fully client-side (in-browser) scoring.

Produces, under docs/model/:
  xgb.onnx, lgb.onnx   - the two raw boosters (uncalibrated P(fraud))
  meta.json            - isotonic calibration breakpoints, ensemble weight,
                         feature order, per-feature medians (for occlusion
                         explanations) and amount-band bin edges.

The browser runs both ONNX models with onnxruntime-web, applies the isotonic
calibration via linear interpolation, and blends 0.55*xgb + 0.45*lgb -- exactly
what src/model.py does in Python. We validate parity here before shipping.

    python scripts/export_onnx.py --data data/creditcard.csv --model artifacts/model.pkl
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

from src.features import build_features, feature_columns  # noqa: E402
from src.model import load_bundle  # noqa: E402

OUT_DIR = "docs/model"
RAW_INPUTS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
AMOUNT_BINS = [-0.01, 1, 10, 50, 200, 1000, 5000, float("inf")]


def _strip_zipmap(model):
    """Replace a ZipMap (seq-of-maps) output with the clean [N,2] prob tensor,
    keeping the output name 'probabilities'. Simplifies reading in JS."""
    import onnx
    from onnx import TensorProto, helper

    g = model.graph
    zips = [nd for nd in g.node if nd.op_type == "ZipMap"]
    if not zips:
        return model
    z = zips[0]
    src, dst = z.input[0], z.output[0]
    g.node.remove(z)
    g.node.append(helper.make_node("Identity", [src], [dst]))
    for i, o in enumerate(g.output):
        if o.name == dst:
            g.output[i].CopyFrom(
                helper.make_tensor_value_info(dst, TensorProto.FLOAT, [None, 2]))
    return model


def _iso(cal):
    """Extract isotonic breakpoints as plain lists for JS interpolation."""
    return {"x": [float(v) for v in cal.X_thresholds_],
            "y": [float(v) for v in cal.y_thresholds_]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--model", default="artifacts/model.pkl")
    args = parser.parse_args()

    from onnxmltools.convert import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort

    os.makedirs(OUT_DIR, exist_ok=True)
    bundle = load_bundle(args.model)
    cols = bundle["feature_cols"]
    n = len(cols)
    w = bundle["weight_xgb"]

    xgb_est = bundle["xgb"].calibrated_classifiers_[0].estimator
    lgb_est = bundle["lgb"].calibrated_classifiers_[0].estimator
    xgb_cal = bundle["xgb"].calibrated_classifiers_[0].calibrators[0]
    lgb_cal = bundle["lgb"].calibrated_classifiers_[0].calibrators[0]

    # Booster carries DataFrame column names ("V14"); the converter needs the
    # positional "f0.." convention. Clear them (column order is preserved).
    try:
        xgb_est.get_booster().feature_names = None
    except Exception:
        pass

    init = [("input", FloatTensorType([None, n]))]
    # zipmap=False -> probabilities come back as a clean [N,2] float tensor
    # (no sequence-of-maps), which is far simpler to read in onnxruntime-web.
    onx_xgb = _strip_zipmap(convert_xgboost(xgb_est, initial_types=init, target_opset=12))
    onx_lgb = _strip_zipmap(convert_lightgbm(lgb_est, initial_types=init, target_opset=12))
    with open(f"{OUT_DIR}/xgb.onnx", "wb") as f:
        f.write(onx_xgb.SerializeToString())
    with open(f"{OUT_DIR}/lgb.onnx", "wb") as f:
        f.write(onx_lgb.SerializeToString())
    print(f"[onnx] wrote xgb.onnx / lgb.onnx ({n} input features)")

    # --- feature medians (for occlusion attribution) from the real data ---
    df = build_features(pd.read_csv(args.data))
    medians = {c: float(df[c].median()) for c in RAW_INPUTS}

    meta = {
        "model_version": bundle["model_version"],
        "feature_cols": cols,
        "raw_inputs": RAW_INPUTS,
        "weight_xgb": float(w),
        "iso_xgb": _iso(xgb_cal),
        "iso_lgb": _iso(lgb_cal),
        "medians": medians,
        "amount_bins": AMOUNT_BINS[:-1] + [1e12],  # JSON-safe (no inf)
        "thresholds": {"review": 0.43, "block": 0.95},
    }
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[onnx] wrote meta.json (iso breakpoints, {len(medians)} medians)")

    # ---------------- parity check vs the Python bundle ----------------
    def interp(cal_x, cal_y, p):
        return float(np.interp(p, cal_x, cal_y))

    sess_x = ort.InferenceSession(f"{OUT_DIR}/xgb.onnx")
    sess_l = ort.InferenceSession(f"{OUT_DIR}/lgb.onnx")
    out_x = [o.name for o in sess_x.get_outputs()]
    out_l = [o.name for o in sess_l.get_outputs()]
    print(f"[onnx] xgb outputs={out_x} lgb outputs={out_l}")

    def prob_out(sess, names, X):
        out_names = [o.name for o in sess.get_outputs()]
        res = sess.run(None, {"input": X.astype(np.float32)})
        by_name = dict(zip(out_names, res))
        arr = np.array(by_name.get("probabilities", res[-1]))
        return arr[:, 1]

    sample = df.sample(n=min(2000, len(df)), random_state=7)
    X = sample[cols].values
    px = prob_out(sess_x, out_x, X)
    pl = prob_out(sess_l, out_l, X)
    onnx_ens = w * np.array([interp(meta["iso_xgb"]["x"], meta["iso_xgb"]["y"], p) for p in px]) \
        + (1 - w) * np.array([interp(meta["iso_lgb"]["x"], meta["iso_lgb"]["y"], p) for p in pl])

    py_x = bundle["xgb"].predict_proba(sample[cols])[:, 1]
    py_l = bundle["lgb"].predict_proba(sample[cols])[:, 1]
    py_ens = w * py_x + (1 - w) * py_l

    max_diff = float(np.max(np.abs(onnx_ens - py_ens)))
    mean_diff = float(np.mean(np.abs(onnx_ens - py_ens)))
    print(f"[parity] over {len(sample)} rows: max|diff|={max_diff:.3e}  mean|diff|={mean_diff:.3e}")
    print("[parity] PASS" if max_diff < 1e-3 else "[parity] WARN - difference larger than 1e-3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
