from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class VectorIndex:
    """A simple cosine-similarity FAISS index over chunk embeddings."""

    def __init__(
        self,
        embed_model: SentenceTransformer,
        embedding_dim: int,
        chunks: list[Chunk] | None = None,
        index: faiss.Index | None = None,
    ) -> None:
        self.embed_model = embed_model
        self.embedding_dim = embedding_dim
        self.chunks: list[Chunk] = chunks or []
        self.index: faiss.Index = index or faiss.IndexFlatIP(embedding_dim)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def build(self, chunks: list[Chunk], show_progress_bar: bool = True) -> None:
        self.chunks = chunks
        if not chunks:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            return

        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=show_progress_bar)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self.ntotal <= 0:
            return []

        query_embedding = self.embed_model.encode([query])
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        results: list[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0], strict=False):
            if int(idx) == -1:
                continue
            results.append(RetrievalResult(chunk=self.chunks[int(idx)], score=float(score)))

        return results

    def retrieve_with_threshold(
        self, query: str, top_k: int = 10, score_threshold: float | None = None
    ) -> list[RetrievalResult]:
        results = self.retrieve(query, top_k=top_k)
        if score_threshold is None:
            return results
        return [r for r in results if r.score > score_threshold]

    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        faiss.write_index(self.index, str(filepath.with_suffix(".faiss")))
        with filepath.with_suffix(".chunks").open("wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, filepath: str | Path, embed_model: SentenceTransformer) -> "VectorIndex":
        filepath = Path(filepath)
        index = faiss.read_index(str(filepath.with_suffix(".faiss")))
        with filepath.with_suffix(".chunks").open("rb") as f:
            chunks: list[Chunk] = pickle.load(f)

        embedding_dim = int(index.d)
        return cls(embed_model=embed_model, embedding_dim=embedding_dim, chunks=chunks, index=index)


def format_retrieval_context(results: Iterable[RetrievalResult]) -> str:
    parts: list[str] = []
    for r in results:
        parts.append(
            f"[Source: {r.chunk.source_file}, Relevance: {r.score:.3f}]\n{r.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)
