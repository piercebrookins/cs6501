from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2

from topic6vlm.config import PERSON_DETECTION_PROMPT
from topic6vlm.llava_client import LlavaClient


def run(model_name: str, interval_seconds: float, camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam index {camera_index}")

    client = LlavaClient(model_name=model_name)

    print("Starting webcam monitor. Press Ctrl+C to stop.")
    with tempfile.TemporaryDirectory(prefix="vlm_webcam_") as tmp_dir:
        frame_path = Path(tmp_dir) / "live_frame.jpg"

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame; retrying...")
                time.sleep(interval_seconds)
                continue

            cv2.imwrite(str(frame_path), frame)
            response = client.chat(PERSON_DETECTION_PROMPT, image_paths=[str(frame_path)])
            has_person = response.strip().upper().startswith("YES")

            stamp = time.strftime("%H:%M:%S")
            if has_person:
                print(f"[{stamp}] INTRUDER ALERT | model_response={response!r}")
            else:
                print(f"[{stamp}] clear | model_response={response!r}")

            time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional realtime webcam VLM monitor")
    parser.add_argument("--model", default="llava")
    parser.add_argument("--interval", type=float, default=8.5)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(model_name=args.model, interval_seconds=args.interval, camera_index=args.camera)
