#!/usr/bin/env python3
"""Simple 4-class SVM trainer for HAR.

Usage:
  python src/train_svm_4class.py --data-dir data --window-sec 2.0 --out models/svm_4class.pkl

Collect CSVs with columns: t_epoch,ax_g,ay_g,az_g,label_id (see src/data_collect.py).
This script windows the data, extracts features, trains SVM, prints accuracy, and saves a bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import segment_windows, extract_features_from_window
import pickle


def _discover_csvs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def _load_csv(csv_path: Path) -> pd.DataFrame:
    # Use comment prefix to ignore the JSON header row.
    df = pd.read_csv(csv_path, comment="#")
    return df


def build_dataset(data_dir: Path, window_sec: float = 2.0, rate_hz: float = 50.0) -> Tuple[pd.DataFrame, np.ndarray, List[str], float]:
    csv_files = _discover_csvs(data_dir)
    window_samples = int(round(window_sec * rate_hz))

    rows: List[pd.Series] = []
    labels: List[int] = []

    for csv_path in csv_files:
        df = _load_csv(csv_path)
        windows = segment_windows(df, window_samples=window_samples, rate_hz=rate_hz)
        for wdf, lid in windows:
            if lid < 0:
                continue
            rows.append(extract_features_from_window(wdf))
            labels.append(int(lid))

    X = pd.DataFrame(rows)
    y = np.asarray(labels, dtype=int)
    feature_names = list(X.columns)
    return X, y, feature_names, float(rate_hz)


def train_and_save_model(data_dir: Path, window_sec: float = 2.0, model_out: Path | None = None) -> Path:
    X, y, feature_names, rate_hz = build_dataset(data_dir=data_dir, window_sec=window_sec)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #scales data
    scaler = StandardScaler() 
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)


    clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}  (n_test={len(y_test)})")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "window_sec": float(window_sec),
        "rate_hz": float(rate_hz),
    }
    if model_out is None:
        model_out = Path("models") / "svm_4class.pkl"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved: {model_out}")
    return model_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 4-class SVM for HAR")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Root directory of CSVs")
    p.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Window size in seconds (default: 2.0)",
    )
    p.add_argument("--out", type=Path, default=Path("models") / "svm_4class.pkl", help="Output model path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_and_save_model(data_dir=args.data_dir, window_sec=args.window_sec, model_out=args.out)


if __name__ == "__main__":
    main()
