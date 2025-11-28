"""Simple feature engineering for accelerometer-based HAR.

Expect columns: 't_epoch', 'ax_g', 'ay_g', 'az_g', 'label_id'.

Simplified behavior:
- Windows are non-overlapping within each contiguous same-label block.
- No time-gap heuristics; just chunk per label in order.
- Minimal feature set for speed and consistency.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def segment_windows(df: pd.DataFrame, window_samples: int, rate_hz: float) -> List[Tuple[pd.DataFrame, int]]:
    df = df[df["label_id"] >= 0].sort_values("t_epoch").reset_index(drop=True)
    if window_samples <= 0 or df.empty:
        return []

    blocks = df["label_id"].ne(df["label_id"].shift(1)).cumsum()
    out: List[Tuple[pd.DataFrame, int]] = []
    for _, seg in df.groupby(blocks, sort=False):
        label = int(seg["label_id"].iloc[0])
        n_full = len(seg) // window_samples
        for i in range(n_full):
            s = i * window_samples
            e = s + window_samples
            out.append((seg.iloc[s:e], label))
    return out


def extract_features_from_window(window_df: pd.DataFrame) -> pd.Series:
    x = window_df["ax_g"].to_numpy(float)
    y = window_df["ay_g"].to_numpy(float)
    z = window_df["az_g"].to_numpy(float)

    mag = np.sqrt(x * x + y * y + z * z)
    sma = np.mean(np.abs(x) + np.abs(y) + np.abs(z))

    feats = {
        "ax_mean": float(np.mean(x)),
        "ay_mean": float(np.mean(y)),
        "az_mean": float(np.mean(z)),
        "ax_std": float(np.std(x)),
        "ay_std": float(np.std(y)),
        "az_std": float(np.std(z)),
        "ax_min": float(np.min(x)),
        "ay_min": float(np.min(y)),
        "az_min": float(np.min(z)),
        "ax_max": float(np.max(x)),
        "ay_max": float(np.max(y)),
        "az_max": float(np.max(z)),
        "mag_mean": float(np.mean(mag)),
        "mag_std": float(np.std(mag)),
        "sma": float(sma),
    }
    order = [
        "ax_mean", "ay_mean", "az_mean",
        "ax_std", "ay_std", "az_std",
        "ax_min", "ay_min", "az_min",
        "ax_max", "ay_max", "az_max",
        "mag_mean", "mag_std", "sma",
    ]
    return pd.Series({k: feats[k] for k in order}, index=order)
