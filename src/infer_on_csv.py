#!/usr/bin/env python3
"""Run a saved HAR model on a single CSV and summarize predictions.

Usage examples:
  python src/infer_on_csv.py --model models/svm_4class.pkl --csv data/2025_11_29_1749/session_50Hz.csv
  python src/infer_on_csv.py --model models/svm_4class.pkl --csv ~/some.csv --window-sec 2.0 --rate-hz 50

Outputs a short summary and writes a predictions CSV next to the input file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from features import segment_windows, extract_features_from_window


def _read_labels_from_header(csv_path: Path) -> List[str] | None:
    try:
        with open(csv_path, "r") as f:
            first = f.readline().strip()
        if first.startswith("#"):
            meta = json.loads(first[1:].strip())
            if isinstance(meta, dict) and "labels" in meta and isinstance(meta["labels"], list):
                return [str(x) for x in meta["labels"]]
    except Exception:
        pass
    return None


def _load_csv(csv_path: Path) -> pd.DataFrame:
    # Use comment prefix to ignore the JSON header row if present.
    return pd.read_csv(csv_path, comment="#")


def predict_on_csv(model_path: Path, csv_path: Path, window_sec: float | None = None, rate_hz: float | None = None, out_csv: Path | None = None) -> Path:
    import pickle

    bundle = pickle.load(open(model_path, "rb"))
    model = bundle["model"]
    scaler = bundle["scaler"]
    b_window_sec = float(bundle.get("window_sec", 2.0))
    b_rate_hz = float(bundle.get("rate_hz", 50.0))

    if window_sec is None:
        window_sec = b_window_sec
    if rate_hz is None:
        rate_hz = b_rate_hz

    window_samples = int(round(float(window_sec) * float(rate_hz)))
    if window_samples <= 0:
        raise ValueError("window_samples must be > 0")

    df = _load_csv(csv_path)
    windows = segment_windows(df, window_samples=window_samples, rate_hz=rate_hz)

    rows: List[pd.Series] = []
    w_meta: List[Tuple[float, float, int]] = []  # (t_start, t_end, src_label)
    for wdf, lid in windows:
        rows.append(extract_features_from_window(wdf))
        t_start = float(wdf["t_epoch"].iloc[0])
        t_end = float(wdf["t_epoch"].iloc[-1])
        w_meta.append((t_start, t_end, int(lid)))

    if not rows:
        raise RuntimeError("No windows produced from CSV; check window size and data length")

    X = pd.DataFrame(rows)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)

    # Map to labels if present
    label_names = _read_labels_from_header(csv_path) or []
    def id_to_name(i: int) -> str:
        if label_names and 0 <= i < len(label_names):
            return label_names[i]
        return str(i)

    # Summary
    counts = Counter(map(int, pred))
    total = len(pred)
    print(f"Predicted {total} windows.")
    for k, v in sorted(counts.items()):
        print(f"  {id_to_name(k)}: {v}")

    # Write per-window predictions
    if out_csv is None:
        out_csv = csv_path.with_name(csv_path.stem + "_pred.csv")
    out = pd.DataFrame({
        "t_start": [m[0] for m in w_meta],
        "t_end": [m[1] for m in w_meta],
        "src_label_id": [m[2] for m in w_meta],
        "pred_label_id": pred.astype(int),
        "pred_label": [id_to_name(int(i)) for i in pred],
    })
    out.to_csv(out_csv, index=False)
    print(f"Saved predictions: {out_csv}")
    return out_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trained HAR model on a CSV")
    p.add_argument("--model", type=Path, required=True, help="Path to pickled model bundle")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV to run inference on")
    p.add_argument("--window-sec", type=float, default=None, help="Window length (defaults to model bundle)")
    p.add_argument("--rate-hz", type=float, default=None, help="Rate (defaults to model bundle)")
    p.add_argument("--out", type=Path, default=None, help="Optional output CSV path for predictions")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    predict_on_csv(model_path=args.model, csv_path=args.csv, window_sec=args.window_sec, rate_hz=args.rate_hz, out_csv=args.out)


if __name__ == "__main__":
    main()

