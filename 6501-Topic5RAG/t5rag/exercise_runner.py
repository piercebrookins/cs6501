from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .chat_writer import ChatWriter
from .corpora import CorpusPaths, get_corpus_paths, validate_pdf_embedded_dir
from .device import DeviceConfig, get_device_config
from .exercise_queries import (
    CONGRESSIONAL_RECORD_QUERIES,
    CROSS_DOC_QUERIES,
    EX1_ALL_QUERIES,
    FAILURE_MODE_QUERIES,
    MODEL_T_QUERIES,
    PHRASING_SENSITIVITY_QUESTION_VARIANTS,
    PROMPT_TEMPLATES,
    SCORE_ANALYSIS_QUERIES,
    UNANSWERABLE_FALSE_PREMISE,
    UNANSWERABLE_OFF_TOPIC,
    UNANSWERABLE_RELATED_BUT_NOT_IN_CORPUS,
)
from .index import RetrievalResult
from .llm import GenerationConfig
from .pipeline import PipelineConfig, RAGPipeline


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    out_dir: Path

    def out_file(self, filename: str) -> Path:
        return self.out_dir / filename

    def index_path(self, name: str) -> Path:
        return self.out_dir / name


def _write_retrieval(session, results: list[RetrievalResult], max_chunk_chars: int = 240) -> None:
    if not results:
        session.write("(no retrieval results)")
        return

    for i, r in enumerate(results, 1):
        excerpt = r.chunk.text.replace("\n", " ")
        if len(excerpt) > max_chunk_chars:
            excerpt = excerpt[:max_chunk_chars] + "..."
        session.write(
            f"[{i}] score={r.score:.4f} source={r.chunk.source_file} chunk={r.chunk.chunk_index} :: {excerpt}"
        )


def _safe_run(session, title: str, fn) -> None:
    session.section(title)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        session.write(f"ERROR: {type(e).__name__}: {e}")


def _build_pipeline(cfg: PipelineConfig, device_cfg: DeviceConfig) -> RAGPipeline:
    """Create a pipeline with sane defaults and env overrides.

    Env overrides (handy when MPS explodes or you want a smaller model):
    - T5RAG_LLM_MODEL
    - T5RAG_LLM_DEVICE (cuda|mps|cpu)
    - T5RAG_EMBED_MODEL
    """

    import torch

    cfg.embedding_model = os.getenv("T5RAG_EMBED_MODEL", cfg.embedding_model)
    cfg.llm_model = os.getenv("T5RAG_LLM_MODEL", cfg.llm_model)

    llm_device = os.getenv("T5RAG_LLM_DEVICE", device_cfg.device)
    llm_dtype = torch.float16 if llm_device == "cuda" else torch.float32

    # SentenceTransformer wants device string; local LLM wrapper wants the same.
    return RAGPipeline.create(
        cfg,
        device=device_cfg.device,
        llm_device=llm_device,
        llm_dtype=llm_dtype,
        load_llm=False,
    )


def _ensure_index_for_corpus(
    *,
    pipeline: RAGPipeline,
    corpus_name: str,
    corpus_folder: Path,
    index_basepath: Path,
    session,
) -> None:
    """Load a saved index if present, otherwise build + save it.

    This is the single biggest speedup for the full assignment run.
    """

    faiss_path = index_basepath.with_suffix(".faiss")
    chunks_path = index_basepath.with_suffix(".chunks")

    if faiss_path.exists() and chunks_path.exists():
        session.write(
            f"Using cached index for {corpus_name}: {faiss_path.name} ({faiss_path.stat().st_size:,} bytes)"
        )
        pipeline.load_index(index_basepath)
        session.kv("index.ntotal", str(pipeline.vector_index.ntotal))
        return

    validate_pdf_embedded_dir(corpus_folder)
    session.write(f"Building index for {corpus_name} (no cache found)...")
    t0 = time.time()
    pipeline.load_and_build_from_folder(corpus_folder)
    t1 = time.time()
    pipeline.save_index(index_basepath)
    session.kv("documents", str(len(pipeline.documents)))
    session.kv("chunks", str(len(pipeline.chunks)))
    session.kv("index.ntotal", str(pipeline.vector_index.ntotal))
    session.kv("build_seconds", f"{(t1 - t0):.2f}")


def run_exercise_0_setup(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_0_setup.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    with writer.session() as s:
        s.section("EXERCISE 0: SETUP")
        s.kv("environment", device_cfg.environment)
        s.kv("device", device_cfg.device)
        s.kv("dtype", str(device_cfg.dtype))
        s.kv("embedding_model", cfg.embedding_model)
        s.kv("llm_model", cfg.llm_model)
        s.kv("chunk_size", str(cfg.chunk_size))
        s.kv("chunk_overlap", str(cfg.chunk_overlap))
        s.blank()

        def build_one(name: str, folder: Path, index_name: str) -> None:
            s.section(f"Building corpus: {name}")

            index_path = paths.index_path(index_name)
            _ensure_index_for_corpus(
                pipeline=pipeline,
                corpus_name=name,
                corpus_folder=folder,
                index_basepath=index_path,
                session=s,
            )

            if pipeline.chunks:
                sample = pipeline.chunks[0]
                s.blank()
                s.write("Sample chunk:")
                s.write(f"source={sample.source_file} chunk={sample.chunk_index}")
                s.write(sample.text[:600])

            s.blank()
            s.write(f"Saved index: {index_path.with_suffix('.faiss')}")
            s.write(f"Saved chunks: {index_path.with_suffix('.chunks')}")

        for name, folder, index_name in [
            ("ModelTService", corpora.model_t, "index_modelt"),
            ("CongressionalRecord", corpora.congressional_record, "index_cr"),
        ]:
            try:
                build_one(name, folder, index_name)
            except Exception as e:  # noqa: BLE001
                s.section(f"Building corpus: {name}")
                s.write(f"SKIPPED: {type(e).__name__}: {e}")


def run_exercise_1_rag_vs_norag(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_1_rag_vs_norag.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    def run_queries(corpus_name: str, folder: Path, queries: list[str], index_name: str) -> None:
        index_path = paths.index_path(index_name)
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name=corpus_name,
            corpus_folder=folder,
            index_basepath=index_path,
            session=s,
        )

        for q in queries:
            s.section(f"{corpus_name} :: {q}")

            t0 = time.time()
            direct = pipeline.direct_query(q)
            t1 = time.time()

            t2 = time.time()
            rag = pipeline.rag_query(q, top_k=5)
            t3 = time.time()

            s.write("NO-RAG answer:")
            s.write(direct)
            s.kv("no_rag_seconds", f"{(t1 - t0):.2f}")
            s.blank()

            s.write("RAG answer:")
            s.write(rag)
            s.kv("rag_seconds", f"{(t3 - t2):.2f}")
            s.blank()

            s.write("Top-5 retrieval (scores + excerpts):")
            _write_retrieval(s, pipeline.retrieve(q, top_k=5))

    with writer.session() as s:
        _safe_run(
            s,
            "EXERCISE 1: RAG vs NO-RAG (Model T)",
            lambda: run_queries("ModelT", corpora.model_t, MODEL_T_QUERIES, "index_modelt"),
        )
        _safe_run(
            s,
            "EXERCISE 1: RAG vs NO-RAG (Congressional Record)",
            lambda: run_queries("CR", corpora.congressional_record, CONGRESSIONAL_RECORD_QUERIES, "index_cr"),
        )

        # Optional subtask: mixed index
        def mixed() -> None:
            s.section("OPTIONAL: Mixed corpora (ModelT + CR) in ONE index")
            pipeline.load_and_build_from_folders([corpora.model_t, corpora.congressional_record])
            s.kv("documents", str(len(pipeline.documents)))
            s.kv("chunks", str(len(pipeline.chunks)))
            s.kv("index.ntotal", str(pipeline.vector_index.ntotal))

            for q in EX1_ALL_QUERIES:
                s.section(f"MIXED :: {q}")
                rag = pipeline.rag_query(q, top_k=5)
                s.write(rag)
                s.write("Top-5 retrieval:")
                _write_retrieval(s, pipeline.retrieve(q, top_k=5))

        _safe_run(s, "EXERCISE 1 OPTIONAL: Mixed corpora", mixed)


def run_exercise_2_gpt4o_mini(paths: RunPaths) -> None:
    out_path = paths.out_file("exercise_2_gpt4o_mini_comparison.txt")
    writer = ChatWriter(out_path)

    with writer.session() as s:
        s.section("EXERCISE 2: GPT-4o Mini comparison")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            s.write("SKIPPED: OPENAI_API_KEY not set.")
            s.write("Set it and re-run to generate this file.")
            return

        try:
            import openai

            client = openai.OpenAI(api_key=api_key)
        except Exception as e:  # noqa: BLE001
            s.write(f"SKIPPED: failed to import/init openai client: {e}")
            return

        def gpt4o_mini_query(question: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
            )
            return response.choices[0].message.content

        for q in EX1_ALL_QUERIES:
            s.section(q)
            t0 = time.time()
            ans = gpt4o_mini_query(q)
            t1 = time.time()
            s.write(ans)
            s.kv("seconds", f"{(t1 - t0):.2f}")

        s.blank()
        s.write(
            "NOTE: Compare this file against `exercise_1_rag_vs_norag.txt` (Qwen no-RAG + RAG)."
        )


def run_exercise_3_frontier_manual(paths: RunPaths) -> None:
    out_path = paths.out_file("exercise_3_frontier_comparison.txt")
    writer = ChatWriter(out_path)

    with writer.session() as s:
        s.section("EXERCISE 3: Frontier chat model comparison (manual)")
        s.write("Manual step (web UI). No file upload. Use the SAME 8 queries from Exercise 1:")
        for q in EX1_ALL_QUERIES:
            s.write(f"- {q}")
        s.blank()
        s.write("Record observations:")
        s.write("- Where general knowledge succeeds")
        s.write("- Where it hallucinates")
        s.write("- Whether it appears to use web search")
        s.write("- Where your local Qwen+RAG is more specific/accurate")


def run_exercise_4_topk(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_4_topk_effect.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    # Pick a few representative queries from the Model T set by default.
    queries = [
        MODEL_T_QUERIES[0],
        MODEL_T_QUERIES[2],
        MODEL_T_QUERIES[3],
    ]

    with writer.session() as s:
        s.section("EXERCISE 4: Effect of top_k retrieval count")
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name="ModelT",
            corpus_folder=corpora.model_t,
            index_basepath=paths.index_path("index_modelt"),
            session=s,
        )

        for q in queries:
            s.section(f"Query: {q}")
            for k in [1, 3, 5, 10, 20]:
                t0 = time.time()
                ans = pipeline.rag_query(q, top_k=k)
                t1 = time.time()
                s.write(f"TOP_K={k} seconds={(t1 - t0):.2f}")
                s.write(ans)
                s.write("retrieval:")
                _write_retrieval(s, pipeline.retrieve(q, top_k=min(k, 10)))
                s.blank()


def run_exercise_5_unanswerable(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_5_unanswerable.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    strict = PROMPT_TEMPLATES["strict"]

    with writer.session() as s:
        s.section("EXERCISE 5: Handling unanswerable questions")
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name="ModelT",
            corpus_folder=corpora.model_t,
            index_basepath=paths.index_path("index_modelt"),
            session=s,
        )

        def run_set(title: str, qs: list[str]) -> None:
            s.section(title)
            for q in qs:
                s.section(q)
                s.write("Baseline prompt:")
                s.write(pipeline.rag_query(q, top_k=5))
                s.write("Strict grounding prompt:")
                s.write(pipeline.rag_query(q, top_k=5, prompt_template=strict))
                s.write("Top-5 retrieval:")
                _write_retrieval(s, pipeline.retrieve(q, top_k=5))

        run_set("Off-topic", UNANSWERABLE_OFF_TOPIC)
        run_set("Related but not in corpus", UNANSWERABLE_RELATED_BUT_NOT_IN_CORPUS)
        run_set("False premise", UNANSWERABLE_FALSE_PREMISE)


def run_exercise_6_phrasing(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_6_phrasing_sensitivity.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    with writer.session() as s:
        s.section("EXERCISE 6: Query phrasing sensitivity")
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name="ModelT",
            corpus_folder=corpora.model_t,
            index_basepath=paths.index_path("index_modelt"),
            session=s,
        )

        for q in PHRASING_SENSITIVITY_QUESTION_VARIANTS:
            s.section(q)
            results = pipeline.retrieve(q, top_k=5)
            _write_retrieval(s, results, max_chunk_chars=400)


def run_exercise_7_overlap(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_7_chunk_overlap.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    queries = [MODEL_T_QUERIES[0], MODEL_T_QUERIES[1], MODEL_T_QUERIES[3]]

    with writer.session() as s:
        s.section("EXERCISE 7: Chunk overlap experiment")
        validate_pdf_embedded_dir(corpora.model_t)
        pipeline.load_and_build_from_folder(corpora.model_t)

        for overlap in [0, 64, 128, 256]:
            s.section(f"overlap={overlap}")
            pipeline.rebuild(chunk_size=512, chunk_overlap=overlap)
            s.kv("chunks", str(len(pipeline.chunks)))
            s.kv("index.ntotal", str(pipeline.vector_index.ntotal))

            for q in queries:
                s.section(q)
                s.write(pipeline.rag_query(q, top_k=5))
                s.write("retrieval:")
                _write_retrieval(s, pipeline.retrieve(q, top_k=5))


def run_exercise_8_chunk_size(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_8_chunk_size.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    queries = [MODEL_T_QUERIES[0], MODEL_T_QUERIES[2], MODEL_T_QUERIES[3], "Brake adjustment procedure", "Cooling system maintenance"]

    with writer.session() as s:
        s.section("EXERCISE 8: Chunk size experiment")
        validate_pdf_embedded_dir(corpora.model_t)
        pipeline.load_and_build_from_folder(corpora.model_t)

        for size in [128, 256, 512, 1024, 2048]:
            overlap = size // 4
            s.section(f"chunk_size={size} chunk_overlap={overlap}")
            pipeline.rebuild(chunk_size=size, chunk_overlap=overlap)
            s.kv("chunks", str(len(pipeline.chunks)))
            s.kv("index.ntotal", str(pipeline.vector_index.ntotal))

            for q in queries[:5]:
                s.section(q)
                s.write(pipeline.rag_query(q, top_k=5))


def run_exercise_9_score_analysis(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_9_score_analysis.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    with writer.session() as s:
        s.section("EXERCISE 9: Retrieval score analysis")
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name="ModelT",
            corpus_folder=corpora.model_t,
            index_basepath=paths.index_path("index_modelt"),
            session=s,
        )

        for q in SCORE_ANALYSIS_QUERIES:
            s.section(f"Query: {q}")
            results = pipeline.retrieve(q, top_k=10)
            _write_retrieval(s, results)

        s.section("Threshold experiment")
        threshold = 0.5
        s.kv("threshold", str(threshold))
        for q in [MODEL_T_QUERIES[0], MODEL_T_QUERIES[3]]:
            s.section(q)
            results = pipeline.vector_index.retrieve_with_threshold(q, top_k=10, score_threshold=threshold)
            _write_retrieval(s, results)
            s.write("Answer w/ thresholded context:")
            ans = pipeline.rag_query(q, top_k=10, score_threshold=threshold)
            s.write(ans)


def run_exercise_10_prompt_templates(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_10_prompt_templates.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    test_queries = MODEL_T_QUERIES

    with writer.session() as s:
        s.section("EXERCISE 10: Prompt template variations")
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name="ModelT",
            corpus_folder=corpora.model_t,
            index_basepath=paths.index_path("index_modelt"),
            session=s,
        )

        for name, template in PROMPT_TEMPLATES.items():
            s.section(f"TEMPLATE: {name}")
            for q in test_queries[:5]:
                s.section(f"Q: {q}")
                ans = pipeline.rag_query(q, top_k=5, prompt_template=template)
                s.write(ans)


def run_exercise_11_failure_modes(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_11_failure_modes.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    with writer.session() as s:
        s.section("EXERCISE 11: Failure mode catalog")
        _ensure_index_for_corpus(
            pipeline=pipeline,
            corpus_name="ModelT",
            corpus_folder=corpora.model_t,
            index_basepath=paths.index_path("index_modelt"),
            session=s,
        )

        for mode, q in FAILURE_MODE_QUERIES:
            s.section(f"{mode} :: {q}")
            s.write("RAG answer:")
            s.write(pipeline.rag_query(q, top_k=5))
            s.write("Top-5 retrieval:")
            _write_retrieval(s, pipeline.retrieve(q, top_k=5))


def run_exercise_12_cross_doc(paths: RunPaths, corpora: CorpusPaths, device_cfg: DeviceConfig) -> None:
    out_path = paths.out_file("exercise_12_cross_document.txt")
    writer = ChatWriter(out_path)

    cfg = PipelineConfig()
    pipeline = _build_pipeline(cfg, device_cfg)

    with writer.session() as s:
        s.section("EXERCISE 12: Cross-document synthesis")

        folders: list[Path] = []
        for label, folder in [
            ("ModelT", corpora.model_t),
            ("CR", corpora.congressional_record),
            ("Learjet", corpora.learjet),
            ("EUAIAct", corpora.eu_ai_act),
        ]:
            try:
                validate_pdf_embedded_dir(folder)
                folders.append(folder)
                s.write(f"Including corpus: {label} -> {folder}")
            except Exception as e:  # noqa: BLE001
                s.write(f"Skipping corpus {label}: {e}")

        if not folders:
            s.write("ERROR: No corpora available for cross-document index.")
            return

        pipeline.load_and_build_from_folders(folders)
        s.kv("documents", str(len(pipeline.documents)))
        s.kv("chunks", str(len(pipeline.chunks)))
        s.kv("index.ntotal", str(pipeline.vector_index.ntotal))

        for q in CROSS_DOC_QUERIES:
            s.section(q)
            for k in [3, 5, 10]:
                s.write(f"top_k={k}")
                ans = pipeline.rag_query(q, top_k=k, generation_cfg=GenerationConfig(max_new_tokens=512, temperature=0.3))
                s.write(ans)
                s.write("retrieval:")
                _write_retrieval(s, pipeline.retrieve(q, top_k=min(k, 10)), max_chunk_chars=280)


def run_all_exercises(repo_root: str | Path = ".") -> None:
    repo_root = Path(repo_root).resolve()
    paths = RunPaths(repo_root=repo_root, out_dir=repo_root / "Topic5RAG")

    device_cfg = get_device_config()

    corpora_error: str | None = None

    # Corpora are required for everything except exercises 2/3.
    try:
        corpora = get_corpus_paths(repo_root)
    except Exception as e:  # noqa: BLE001
        corpora = None
        corpora_error = f"{type(e).__name__}: {e}"

    # Exercise 0/1/4/5/6/7/8/9/10/11/12 require corpora.
    if corpora is None:
        missing_msg = "SKIPPED: Corpora not found. Add `Corpora/` or `Corpora.zip` to repo root, then re-run."
        if corpora_error:
            missing_msg += f" Reason: {corpora_error}"
        for fname in [
            "exercise_0_setup.txt",
            "exercise_1_rag_vs_norag.txt",
            "exercise_4_topk_effect.txt",
            "exercise_5_unanswerable.txt",
            "exercise_6_phrasing_sensitivity.txt",
            "exercise_7_chunk_overlap.txt",
            "exercise_8_chunk_size.txt",
            "exercise_9_score_analysis.txt",
            "exercise_10_prompt_templates.txt",
            "exercise_11_failure_modes.txt",
            "exercise_12_cross_document.txt",
        ]:
            writer = ChatWriter(paths.out_file(fname))
            with writer.session() as s:
                s.section(fname.replace("_", " "))
                s.write(missing_msg)

        # Still do 2 and 3.
        run_exercise_2_gpt4o_mini(paths)
        run_exercise_3_frontier_manual(paths)
        return

    run_exercise_0_setup(paths, corpora, device_cfg)
    run_exercise_1_rag_vs_norag(paths, corpora, device_cfg)
    run_exercise_2_gpt4o_mini(paths)
    run_exercise_3_frontier_manual(paths)
    run_exercise_4_topk(paths, corpora, device_cfg)
    run_exercise_5_unanswerable(paths, corpora, device_cfg)
    run_exercise_6_phrasing(paths, corpora, device_cfg)
    run_exercise_7_overlap(paths, corpora, device_cfg)
    run_exercise_8_chunk_size(paths, corpora, device_cfg)
    run_exercise_9_score_analysis(paths, corpora, device_cfg)
    run_exercise_10_prompt_templates(paths, corpora, device_cfg)
    run_exercise_11_failure_modes(paths, corpora, device_cfg)
    run_exercise_12_cross_doc(paths, corpora, device_cfg)
