from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sentence_transformers import SentenceTransformer

from .chunking import Chunk, chunk_documents
from .documents import Document, load_documents
from .index import RetrievalResult, VectorIndex, format_retrieval_context
from .llm import GenerationConfig, LocalLLM


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question based ONLY on the information in the context above
- If the context doesn't contain enough information to answer, say so
- Quote relevant parts of the context to support your answer
- Be concise and direct

ANSWER:"""


@dataclass
class PipelineConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    chunk_size: int = 512
    chunk_overlap: int = 128


class RAGPipeline:
    """Notebook-style RAG pipeline, but as a proper object.

    Key improvement over the notebook: the local LLM is loaded lazily so you can
    build indexes (which can already take a while) without also downloading a
    multi-GB model first.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        embed_model: SentenceTransformer,
        vector_index: VectorIndex,
        *,
        llm: LocalLLM | None = None,
        llm_device: str,
        llm_dtype: "object",
        documents: list[Document] | None = None,
        chunks: list[Chunk] | None = None,
    ) -> None:
        self.cfg = cfg
        self.embed_model = embed_model
        self.vector_index = vector_index

        self._llm: LocalLLM | None = llm
        self._llm_device = llm_device
        self._llm_dtype = llm_dtype

        self.documents: list[Document] = documents or []
        self.chunks: list[Chunk] = chunks or []

    @classmethod
    def create(
        cls,
        cfg: PipelineConfig,
        *,
        device: str,
        llm_device: str,
        llm_dtype: "object",
        load_llm: bool = False,
    ) -> "RAGPipeline":
        embed_model = SentenceTransformer(cfg.embedding_model, device=device)
        embedding_dim = embed_model.get_sentence_embedding_dimension()
        vector_index = VectorIndex(embed_model=embed_model, embedding_dim=embedding_dim)

        llm = LocalLLM.load(cfg.llm_model, device=llm_device, dtype=llm_dtype) if load_llm else None

        return cls(
            cfg=cfg,
            embed_model=embed_model,
            vector_index=vector_index,
            llm=llm,
            llm_device=llm_device,
            llm_dtype=llm_dtype,
        )

    @property
    def llm(self) -> LocalLLM:
        if self._llm is None:
            raise RuntimeError(
                "LLM not loaded. Call `ensure_llm_loaded()` before calling direct_query/rag_query."
            )
        return self._llm

    def ensure_llm_loaded(self) -> None:
        if self._llm is None:
            self._llm = LocalLLM.load(
                self.cfg.llm_model, device=self._llm_device, dtype=self._llm_dtype
            )

    def load_and_build_from_folder(self, folder: str | Path) -> None:
        self.documents = load_documents(folder)
        self.rebuild(self.cfg.chunk_size, self.cfg.chunk_overlap)

    def load_and_build_from_folders(self, folders: Iterable[str | Path]) -> None:
        docs: list[Document] = []
        for f in folders:
            docs.extend(load_documents(f))
        self.documents = docs
        self.rebuild(self.cfg.chunk_size, self.cfg.chunk_overlap)

    def rebuild(self, chunk_size: int, chunk_overlap: int) -> None:
        self.cfg.chunk_size = chunk_size
        self.cfg.chunk_overlap = chunk_overlap
        self.chunks = chunk_documents(
            self.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.vector_index.build(self.chunks, show_progress_bar=True)

    def save_index(self, filepath: str | Path) -> None:
        self.vector_index.save(filepath)

    def load_index(self, filepath: str | Path) -> None:
        self.vector_index = VectorIndex.load(filepath, embed_model=self.embed_model)
        self.chunks = list(self.vector_index.chunks)

    def direct_query(self, question: str, max_new_tokens: int = 512) -> str:
        self.ensure_llm_loaded()
        prompt = f"""Answer this question:
{question}

Answer:"""
        return self.llm.generate(prompt, cfg=GenerationConfig(max_new_tokens=max_new_tokens))

    def rag_query(
        self,
        question: str,
        top_k: int = 5,
        *,
        show_context: bool = False,
        prompt_template: str | None = None,
        generation_cfg: GenerationConfig | None = None,
        score_threshold: float | None = None,
    ) -> str:
        self.ensure_llm_loaded()

        results = (
            self.vector_index.retrieve_with_threshold(
                question, top_k=top_k, score_threshold=score_threshold
            )
            if score_threshold is not None
            else self.vector_index.retrieve(question, top_k=top_k)
        )

        context = format_retrieval_context(results)
        if show_context:
            print("=" * 60)
            print("RETRIEVED CONTEXT:")
            print("=" * 60)
            print(context)
            print("=" * 60 + "\n")

        template = prompt_template if prompt_template is not None else PROMPT_TEMPLATE
        prompt = template.format(context=context, question=question)

        return self.llm.generate(prompt, cfg=generation_cfg)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.vector_index.retrieve(query, top_k=top_k)
