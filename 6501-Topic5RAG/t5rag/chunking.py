from __future__ import annotations

from dataclasses import dataclass

from .documents import Document


@dataclass(frozen=True)
class Chunk:
    """A chunk of text with metadata for tracing back to source."""

    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Mirrors the notebook implementation (character-based). We try to break at
    paragraph or sentence boundaries to reduce mid-thought cuts.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            para_break = text.rfind("\n\n", start + chunk_size // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                sentence_break = text.rfind(". ", start + chunk_size // 2, end)
                if sentence_break != -1:
                    end = sentence_break + 2

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    text=chunk_str,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_index += 1

        start = end - chunk_overlap
        if chunks and start <= chunks[-1].start_char:
            start = end  # safety: ensure progress

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_text(
                doc.content,
                source_file=doc.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks
