#!/usr/bin/env python3

"""Live HAR prediction on Raspberry Pi using a saved model.

Requirements:
  pip install sense-hat numpy pandas scikit-learn

Usage:
  python src/live_predict_pi.py --model models/svm_4class.pkl [--window-sec 2.0] [--stride-sec 1.0]

Displays predicted class on the Sense HAT LEDs (S/W/R/F) and prints to console.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import deque, Counter
from typing import Deque, Dict, Tuple

import numpy as np
import pandas as pd
from sense_hat import SenseHat

from features import extract_features_from_window


LABELS = ["sitting", "walking", "running", "falling"]
LABEL_TO_LETTER: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "sitting": ("S", (255, 120, 120)),
    "walking": ("W", (120, 255, 120)),
    "running": ("R", (120, 120, 255)),
    "falling": ("F", (255, 255, 255)),
}


def _show_label(sense: SenseHat, label_idx: int) -> None:
    if 0 <= label_idx < len(LABELS):
        name = LABELS[label_idx]
        letter, color = LABEL_TO_LETTER.get(name, (str(label_idx), (255, 255, 255)))
        sense.show_letter(letter, text_colour=color)
    else:
        sense.show_letter("?", text_colour=(255, 255, 255))


def _predict_window(bundle, window_df: pd.DataFrame) -> int:
    X = extract_features_from_window(window_df)
    Xs = bundle["scaler"].transform(pd.DataFrame([X]))
    pred = int(bundle["model"].predict(Xs)[0])
    return pred


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live HAR prediction on Sense HAT")
    p.add_argument("--model", required=True, help="Path to pickled model bundle")
    p.add_argument("--window-sec", type=float, default=None, help="Window seconds (defaults to model bundle)")
    p.add_argument("--stride-sec", type=float, default=None, help="Stride seconds between predictions (default: window_sec/2)")
    p.add_argument("--smooth", type=int, default=3, help="Majority vote over last N predictions (default: 3)")
    p.add_argument("--rotate-deg", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate LED letters (0/90/180/270). Use 180 for upside down.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    rate_hz = float(bundle.get("rate_hz", 50.0))
    window_sec = float(args.window_sec if args.window_sec is not None else bundle.get("window_sec", 2.0))
    stride_sec = float(args.stride_sec if args.stride_sec is not None else max(0.2, window_sec / 2.0))

    window_samples = max(1, int(round(window_sec * rate_hz)))
    stride_samples = max(1, int(round(stride_sec * rate_hz)))

    sense = SenseHat()
    sense.clear()
    sense.set_imu_config(False, False, True)  # accelerometer only
    # Rotate LED matrix so letters appear with desired orientation
    try:
        sense.set_rotation(int(args.rotate_deg))
    except Exception:
        pass

    buf: Deque[Tuple[float, float, float]] = deque(maxlen=window_samples)
    pred_hist: Deque[int] = deque(maxlen=max(1, int(args.smooth)))

    print(f"[START] rate={rate_hz}Hz window={window_sec}s ({window_samples} samples) stride={stride_sec}s ({stride_samples} samples)")
    last_pred_idx = -1

    try:
        n = 0
        while True:
            raw = sense.get_accelerometer_raw()
            ax = float(raw.get("x", 0.0))
            ay = float(raw.get("y", 0.0))
            az = float(raw.get("z", 0.0))
            buf.append((ax, ay, az))
            n += 1

            if len(buf) >= window_samples and (n % stride_samples) == 0:
                wdf = pd.DataFrame(buf, columns=["ax_g", "ay_g", "az_g"])
                # add dummy label/time columns to satisfy feature function expectations
                wdf.insert(0, "t_epoch", np.arange(len(wdf), dtype=float))
                wdf["label_id"] = 0
                pred = _predict_window(bundle, wdf)
                pred_hist.append(pred)
                # smooth
                mode = Counter(pred_hist).most_common(1)[0][0]
                if mode != last_pred_idx:
                    last_pred_idx = mode
                    _show_label(sense, mode)
                print(f"pred={pred} smooth={mode}")

            # pace loop to ~rate_hz
            time.sleep(max(0.0, (1.0 / rate_hz)))
    except KeyboardInterrupt:
        pass
    finally:
        sense.clear()
        print("[DONE]")


if __name__ == "__main__":
    main()
