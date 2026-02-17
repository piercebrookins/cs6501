from __future__ import annotations

import argparse
from dataclasses import dataclass

from topic6vlm.config import PERSON_DETECTION_PROMPT, VLMConfig
from topic6vlm.llava_client import LlavaClient
from topic6vlm.video_utils import FrameSample, extract_frames_every_n_seconds


@dataclass(frozen=True)
class DetectionResult:
    timestamp_seconds: float
    person_present: bool
    raw_response: str
    frame_path: str


def _is_yes(response: str) -> bool:
    normalized = response.strip().upper()
    return normalized.startswith("YES")


def classify_frames(samples: list[FrameSample], model_name: str) -> list[DetectionResult]:
    client = LlavaClient(model_name=model_name)
    results: list[DetectionResult] = []

    for sample in samples:
        response = client.chat(
            prompt=PERSON_DETECTION_PROMPT,
            image_paths=[sample.path],
        )
        results.append(
            DetectionResult(
                timestamp_seconds=sample.timestamp_seconds,
                person_present=_is_yes(response),
                raw_response=response,
                frame_path=sample.path,
            )
        )

    return results


def summarize_entry_exit(results: list[DetectionResult]) -> tuple[list[float], list[float]]:
    entries: list[float] = []
    exits: list[float] = []

    prev_present = False
    for result in results:
        if result.person_present and not prev_present:
            entries.append(result.timestamp_seconds)
        elif not result.person_present and prev_present:
            exits.append(result.timestamp_seconds)
        prev_present = result.person_present

    return entries, exits


def seconds_to_mmss(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def run(args: argparse.Namespace) -> None:
    cfg = VLMConfig(model_name=args.model, frame_interval_seconds=args.interval)
    samples = extract_frames_every_n_seconds(
        video_path=args.video,
        output_dir=args.frames_dir,
        interval_seconds=cfg.frame_interval_seconds,
    )

    if not samples:
        raise ValueError("No frames extracted. Check video path and interval.")

    results = classify_frames(samples, model_name=cfg.model_name)
    entries, exits = summarize_entry_exit(results)

    print("\nFrame-by-frame detection:")
    for r in results:
        print(
            f"- {seconds_to_mmss(r.timestamp_seconds)} | "
            f"person={r.person_present} | response={r.raw_response!r}"
        )

    print("\nSummary")
    print("-------")
    if entries:
        print("Entry times:", ", ".join(seconds_to_mmss(t) for t in entries))
    else:
        print("Entry times: none")

    if exits:
        print("Exit times:", ", ".join(seconds_to_mmss(t) for t in exits))
    else:
        print("Exit times: none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 2: Detect when a person enters/exits a scene using LLaVA"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--frames-dir",
        default="frames_out",
        help="Directory where extracted frames are saved",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Frame sampling interval in seconds",
    )
    parser.add_argument(
        "--model",
        default="llava",
        help="Ollama model name (default: llava)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
