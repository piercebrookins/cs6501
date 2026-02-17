from dataclasses import dataclass


@dataclass(frozen=True)
class VLMConfig:
    """Centralized config for all VLM apps."""

    model_name: str = "llava"
    image_max_width: int = 1024
    image_quality: int = 85
    frame_interval_seconds: float = 2.0


PERSON_DETECTION_PROMPT = (
    "You are a strict visual detector. "
    "Answer with ONLY YES or NO. "
    "Question: Is there at least one person visible in this image?"
)
