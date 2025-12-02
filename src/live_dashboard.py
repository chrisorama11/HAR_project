#!/usr/bin/env python3
"""Tiny Tkinter dashboard that displays live HAR predictions sent via UDP.

Run the Sense HAT inference publisher:
  python src/live_predict_pi.py --model models/svm_4class.pkl --broadcast-udp 127.0.0.1:5050

Then launch this dashboard (on the Pi or forwarded with ssh -X):
  python src/live_dashboard.py --listen 0.0.0.0:5050
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, Tuple


DEFAULT_LABELS = ["sitting", "walking", "running", "falling"]


@dataclass
class PredictionState:
    smooth_label_id: int | None = None
    smooth_label_name: str = "Waiting…"
    raw_label_id: int | None = None
    raw_label_name: str = ""
    updated_ts: float = 0.0
    window_sec: float = 0.0
    stride_sec: float = 0.0
    smooth_window: int = 0


def _parse_listen(value: str) -> Tuple[str, int]:
    host, sep, port_str = value.rpartition(":")
    if not sep:
        raise argparse.ArgumentTypeError("Expected host:port")
    host = host or "0.0.0.0"
    port = int(port_str)
    if not (0 < port <= 65535):
        raise argparse.ArgumentTypeError("Port out of range")
    return host, port


def _build_ui_root(title: str) -> tk.Tk:
    root = tk.Tk()
    root.title(title)
    root.geometry("420x260")
    root.configure(bg="#121212")
    return root


class Dashboard:
    def __init__(self, listen: Tuple[str, int], labels: Dict[int, str]):
        self.listen = listen
        self.labels = labels
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(listen)
        self.sock.setblocking(False)

        self.state = PredictionState()
        self.root = _build_ui_root("HAR Live Dashboard")

        self.big_label = tk.Label(
            self.root,
            text=self.state.smooth_label_name,
            font=("Helvetica", 48, "bold"),
            fg="#00e676",
            bg="#121212",
        )
        self.big_label.pack(pady=(30, 10))

        self.detail_label = tk.Label(
            self.root,
            text="Waiting for data",
            font=("Helvetica", 16),
            fg="#e0e0e0",
            bg="#121212",
        )
        self.detail_label.pack()

        self.meta_label = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 12),
            fg="#9e9e9e",
            bg="#121212",
        )
        self.meta_label.pack(pady=(20, 0))

        self.status_label = tk.Label(
            self.root,
            text=f"Listening on {listen[0]}:{listen[1]}",
            font=("Helvetica", 10),
            fg="#757575",
            bg="#121212",
        )
        self.status_label.pack(side=tk.BOTTOM, pady=10)

        self.root.after(100, self._poll)

    def _update_state(self, payload: dict) -> None:
        smooth_id = payload.get("smooth_label_id")
        smooth_name = payload.get("smooth_label_name")
        raw_id = payload.get("raw_label_id")
        raw_name = payload.get("raw_label_name")

        if smooth_name is None and smooth_id is not None:
            smooth_name = self.labels.get(int(smooth_id), str(smooth_id))
        if raw_name is None and raw_id is not None:
            raw_name = self.labels.get(int(raw_id), str(raw_id))

        self.state = PredictionState(
            smooth_label_id=smooth_id,
            smooth_label_name=str(smooth_name) if smooth_name else "?",
            raw_label_id=raw_id,
            raw_label_name=str(raw_name) if raw_name else "?",
            updated_ts=float(payload.get("timestamp", time.time())),
            window_sec=float(payload.get("window_sec", 0.0)),
            stride_sec=float(payload.get("stride_sec", 0.0)),
            smooth_window=int(payload.get("smooth_window", 0)),
        )

    def _poll(self) -> None:
        updated = False
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            self._update_state(payload)
            updated = True
        if updated:
            self._render()
        self.root.after(100, self._poll)

    def _render(self) -> None:
        self.big_label.configure(text=self.state.smooth_label_name.upper())
        raw_txt = f"Raw: {self.state.raw_label_name.upper()}"
        updated_delta = time.time() - self.state.updated_ts
        self.detail_label.configure(text=f"{raw_txt}    •   Updated {updated_delta:.1f}s ago")
        self.meta_label.configure(
            text=(
                f"Window {self.state.window_sec:.2f}s   |   Stride {self.state.stride_sec:.2f}s"
                f"   |   Smooth N={self.state.smooth_window}"
            )
        )

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.sock.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live HAR dashboard (Tkinter)")
    p.add_argument("--listen", type=_parse_listen, default=("0.0.0.0", 5050), help="Address to bind, e.g., 0.0.0.0:5050")
    p.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label names (index order). Default: sitting,walking,running,falling",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    label_map = {i: name.strip() for i, name in enumerate(args.labels.split(",")) if name.strip()}
    dash = Dashboard(listen=args.listen, labels=label_map)
    dash.run()


if __name__ == "__main__":
    main()

