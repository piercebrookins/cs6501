"""
Document chunker — sliding-window with smart sentence-boundary breaks.
"""

from typing import Any, Dict, List, Optional, Tuple


class DocumentChunker:
    """
    Splits documents into overlapping text chunks.

    Uses a sliding window with sentence-boundary snapping so chunks
    don't cut words/sentences in half (when possible).
    """

    BREAK_DELIMITERS = (". ", "? ", "! ", "\n", " ")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Chunk a list of document dicts.

        Returns:
            (chunks, metadata) — parallel lists.
        """
        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        all_chunks: List[str] = []
        all_meta: List[Dict[str, Any]] = []

        for doc in documents:
            text = doc["text"]
            source = doc["source"]
            page = doc.get("page", 1)
            c, m = self._chunk_text(text, source, page, size, overlap)
            all_chunks.extend(c)
            all_meta.extend(m)

        print(f"Created {len(all_chunks)} chunks (size={size}, overlap={overlap})")
        return all_chunks, all_meta

    # ------------------------------------------------------------------
    def _chunk_text(
        self,
        text: str,
        source: str,
        page: int,
        size: int,
        overlap: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        chunks: List[str] = []
        meta: List[Dict[str, Any]] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + size, len(text))

            # Snap to sentence boundary when not at end-of-text
            if end < len(text):
                end = self._snap_boundary(text, start, end, size)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
                meta.append({
                    "source": source,
                    "page": page,
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                })
                idx += 1

            start += size - overlap
            if start <= 0 or (overlap >= size and end >= len(text)):
                break

        return chunks, meta

    # ------------------------------------------------------------------
    @classmethod
    def _snap_boundary(cls, text: str, start: int, end: int, size: int) -> int:
        """Try to break at the nearest sentence delimiter."""
        min_pos = start + size // 2  # don't snap below half the window
        for delim in cls.BREAK_DELIMITERS:
            pos = text.rfind(delim, start, end)
            if pos != -1 and pos >= min_pos:
                return pos + len(delim)
        return end
