#!/usr/bin/env python3
import io
import time
import threading
import json
from datetime import datetime, timezone
from pathlib import Path
from sense_hat import SenseHat

# =======================
# Config
# =========================
RATE_HZ   = 50.0
DT        = 1.0 / RATE_HZ
DATA_ROOT = Path("/home/pi/Desktop/data")

# Labels / IDs
LABELS  = {"sitting": 0, "walking": 1, "running": 2, "falling": 3}
IDLE_ID = -1

# Joystick mapping (latched)
DIR_TO_LABEL = {"up": "sitting", "right": "walking", "left": "running", "down": "falling"}  # CENTER will toggle idle
DEBOUNCE_S   = 0.15
LETTER_FLASH_S = 0.08
LABEL_DISPLAY = {
    "sitting": ("S", (255, 120, 120)),
    "walking": ("W", (120, 255, 120)),
    "running": ("R", (120, 120, 255)),
    "falling": ("F", (255, 255, 255)),
}

# =========================
# Sense HAT (perf-minded)
# =========================
sense = SenseHat()
sense.clear()  # turn off LEDs entirely during capture
# Enable accelerometer only
sense.set_imu_config(False, False, True)

def _flash_letter(letter: str, color):
    """Clear, briefly flash, then hold the given letter."""
    sense.clear()
    if LETTER_FLASH_S > 0:
        time.sleep(LETTER_FLASH_S)
    sense.show_letter(letter, text_colour=color)

def display_activity(label_name):
    """Update the LED matrix for the given activity name (or clear for idle)."""
    if not label_name:
        sense.clear()
        return
    letter, color = LABEL_DISPLAY.get(label_name, ("?", (255, 255, 255)))
    _flash_letter(letter, color)

# =========================
# Paths & metadata
# =========================
now_local   = datetime.now()
session_dir = DATA_ROOT / now_local.strftime("%Y_%m_%d_%H%M")
session_dir.mkdir(parents=True, exist_ok=True)
csv_path = session_dir / "session_50Hz.csv"

meta = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "rate_hz": RATE_HZ,
    "mount": "waist",
    "axes": "x=forward,y=left,z=up",
    "labels": ["sitting", "walking", "running", "falling"],
    "idle_id": IDLE_ID,
    "ui": "UP=sitting, RIGHT=walking, LEFT=running, DOWN=falling, CENTER=idle (press), Ctrl-C=end",
    "notes": "Accelerometer-only; LEDs off; joystick in separate thread; batched writes; 50Hz sampling",
}

# =========================
# Shared state (thread-safe)
# =========================
_label_id     = IDLE_ID
_running      = True
_last_press_t = 0.0
_lock         = threading.Lock()

def set_label(new_id: int):
    global _label_id
    with _lock:
        _label_id = new_id

def get_label() -> int:
    with _lock:
        return _label_id

def stop_running():
    global _running
    with _lock:
        _running = False

def is_running() -> bool:
    with _lock:
        return _running

# =========================
# Joystick thread
# =========================
def joystick_worker():
    global _last_press_t
    while is_running():
        events = sense.stick.get_events()
        if not events:
            time.sleep(0.02)  # small yield
            continue
        nowp = time.perf_counter()
        for e in events:
            if e.action != "pressed":
                continue
            if (nowp - _last_press_t) < DEBOUNCE_S:
                continue
            _last_press_t = nowp

            if e.direction == "middle":
                # Map CENTER press to IDLE: stop labeling, clear LEDs
                set_label(IDLE_ID)
                display_activity(None)
                print("IDLE")
                continue
            # if e.direction == "down":
            #     set_label(IDLE_ID)
            #     display_activity(None)
            #     print("IDLE")
            #     continue
            if e.direction in DIR_TO_LABEL:
                label_name = DIR_TO_LABEL[e.direction]  # 'sitting', 'walking', 'running', 'falling'
                set_label(LABELS[label_name])
                display_activity(label_name)
                print(label_name.upper())
# =========================
# Capture
# =========================
def main():
    jt = threading.Thread(target=joystick_worker, daemon=True)
    jt.start()

    # Startup banner: make console output explicit about what's being recorded
    print(f"[START] Session dir: {session_dir}")
    print(f"[START] Target rate: {RATE_HZ} Hz")
    print(f"[START] Writing CSV: {csv_path}")
    print("[START] Columns: t_epoch, ax_g, ay_g, az_g, label_id")
    print(f"[START] Joystick: {meta['ui']}")
    print("[HINT] Press a direction to set label; CENTER sets IDLE; Ctrl-C to stop\n")

    with open(csv_path, "w", buffering=1024*1024) as f:  # large OS buffer
        f.write("# " + json.dumps(meta) + "\n")
        f.write("t_epoch,ax_g,ay_g,az_g,label_id\n")

        buf = io.StringIO()
        batch_size = 50   # ~1s per flush at 50 Hz CHANGE
        in_batch   = 0

        n_samples = 0
        t0   = time.perf_counter()
        tlast = t0

        try:
            while is_running():
                # Read accelerometer (accelerometer-only mode is already set)
                raw = sense.get_accelerometer_raw()
                ax = float(raw.get("x", 0.0))
                ay = float(raw.get("y", 0.0))
                az = float(raw.get("z", 0.0))
                lid = get_label()

                # Buffer line 
                buf.write(f"{time.time():.6f},{ax:.6f},{ay:.6f},{az:.6f},{lid}\n")
                in_batch += 1
                n_samples += 1

                if in_batch >= batch_size:
                    f.write(buf.getvalue())
                    buf.seek(0); buf.truncate(0)
                    in_batch = 0

                # Pace to 50 Hz
                ttarget = tlast + DT
                nowp = time.perf_counter()
                sleep_s = ttarget - nowp
                if sleep_s > 0:
                    time.sleep(sleep_s)
                    tlast = ttarget
                else:
                    # overrun; catch up next cycle
                    tlast = time.perf_counter()
        except KeyboardInterrupt:
            pass
        finally:
            if in_batch > 0:
                f.write(buf.getvalue())

    elapsed  = max(1e-9, time.perf_counter() - t0)
    achieved = n_samples / elapsed
    print(f"[DONE] Saved: {csv_path}")
    print(f"[STATS] Samples: {n_samples}  Duration: {elapsed:.2f}s  Achieved rate: {achieved:.2f} Hz")

if __name__ == "__main__":
    main()
