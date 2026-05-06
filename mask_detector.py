# mask_detector.py
# Written by Zain Rashid
# This script uses a Teachable Machine model to detect whether a person is wearing a mask via webcam.

import os
import sys
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _DW

# ------------- Settings (You can change these if needed) ------------------
MODEL_PATH = "keras_model.h5"           # Pre-trained Teachable Machine model
LABELS_PATH = "labels.txt"              # Labels (e.g., "Mask", "No Mask")
INPUT_SIZE = (224, 224)                 # Input size for model
CONF_SHOW_DECIMALS = 1                  # Decimal places for confidence %
SMOOTHING = 5                           # Moving average smoothing (set 1 to disable)
CAM_INDEX_DEFAULT = 0                   # Default camera (0 is built-in webcam)

# --------- Fix for older keras_model.h5 export compatibility ----------
class PatchedDepthwiseConv2D(_DW):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# -------------------- Load class labels --------------------
def load_labels(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
            return labels
    return ["Mask", "No Mask"]

# ------------- Preprocess image/frame before model prediction -------------
def preprocess_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

# --------- Detect mask status and assign color -----------------
def pick_color_is_mask(label_text: str):
    l = label_text.strip().lower()
    is_mask = ("mask" in l) and ("no" not in l) and ("without" not in l)
    color = (0, 180, 0) if is_mask else (0, 0, 220)
    return is_mask, color

# ------------------ Draw results on video frame -------------------
def overlay_banner(frame, label, conf, fps=None):
    h, w = frame.shape[:2]
    banner_h = 50
    is_mask, color = pick_color_is_mask(label)
    cv2.rectangle(frame, (0, 0), (w, banner_h), (30, 30, 30), -1)
    msg = f"{label}: {conf*100:.{CONF_SHOW_DECIMALS}f}%"
    if fps is not None:
        msg += f"   |   FPS: {fps:.1f}"
    cv2.putText(frame, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "MASK" if is_mask else "NO MASK", (10, banner_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return frame, is_mask

# ------------------ Run mask detection using webcam -------------------
def run_webcam(model, labels, cam_index=CAM_INDEX_DEFAULT):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}.")
        return

    prob_buffer = []
    prev = time.time()
    print("[INFO] Press 'q' to quit webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame.")
            break

        x = preprocess_frame(frame)
        preds = model.predict(x, verbose=0)[0]
        prob_buffer.append(preds)
        if len(prob_buffer) > SMOOTHING:
            prob_buffer.pop(0)
        avg = np.mean(prob_buffer, axis=0)

        idx = int(np.argmax(avg))
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        conf = float(avg[idx])

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6); prev = now

        out, _ = overlay_banner(frame, label, conf, fps=fps)
        cv2.imshow("Mask Detector - Webcam", out)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ Main Execution Starts Here -------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        sys.exit(1)

    print("[INFO] Loading model...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D},
    )
    labels = load_labels(LABELS_PATH)
    print(f"[INFO] Labels: {labels}")

    # Prompt user to start webcam
    print("\n[INFO] Press 1 to start webcam for mask detection.")
    choice = input("Enter 1: ").strip()

    if choice == "1":
        cam_idx = input(f"Enter camera index (default {CAM_INDEX_DEFAULT}): ").strip()
        cam_idx = CAM_INDEX_DEFAULT if cam_idx == "" else int(cam_idx)
        run_webcam(model, labels, cam_index=cam_idx)
    else:
        print("[ERROR] Invalid input. Please restart and press 1.")

if __name__ == "__main__":
    main()