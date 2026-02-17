from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class FrameSample:
    path: str
    timestamp_seconds: float


def extract_frames_every_n_seconds(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 2.0,
) -> list[FrameSample]:
    """Extract JPEG frames from a video on a fixed time interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Unable to read FPS from video.")

    frame_interval = max(1, int(round(fps * interval_seconds)))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[FrameSample] = []
    frame_num = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_num % frame_interval == 0:
            timestamp = frame_num / fps
            frame_path = out_dir / f"frame_{len(samples):04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            samples.append(FrameSample(path=str(frame_path), timestamp_seconds=timestamp))

        frame_num += 1

    cap.release()
    return samples
