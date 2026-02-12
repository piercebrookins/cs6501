from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


def _prefix_lines(text: str, prefix: str) -> str:
    # Preserve trailing newline behavior consistently.
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(f"{prefix}{line}" for line in lines)


@dataclass
class ChatWriter:
    path: Path
    prefix: str = "chat: "

    def open(self) -> TextIO:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path.open("w", encoding="utf-8")

    @contextmanager
    def session(self) -> Iterator["ChatWriterSession"]:
        with self.open() as f:
            yield ChatWriterSession(f=f, prefix=self.prefix)


@dataclass
class ChatWriterSession:
    f: TextIO
    prefix: str

    def write(self, text: str = "") -> None:
        if text == "":
            self.f.write(f"{self.prefix}\n")
            self.f.flush()
            return

        self.f.write(_prefix_lines(text, self.prefix))
        self.f.write("\n")
        self.f.flush()

    def section(self, title: str) -> None:
        self.write("=" * 80)
        self.write(title)
        self.write("=" * 80)

    def kv(self, key: str, value: str) -> None:
        self.write(f"{key}: {value}")

    def blank(self) -> None:
        self.write("")
