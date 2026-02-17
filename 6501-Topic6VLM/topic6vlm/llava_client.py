from __future__ import annotations

import ollama


class LlavaClient:
    """Thin wrapper around Ollama chat for image+text prompts."""

    def __init__(self, model_name: str = "llava") -> None:
        self.model_name = model_name

    def chat(self, prompt: str, image_paths: list[str] | None = None) -> str:
        image_paths = image_paths or []
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_paths,
                }
            ],
        )
        return response["message"]["content"].strip()
