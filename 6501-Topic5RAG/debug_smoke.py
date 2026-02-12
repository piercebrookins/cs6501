from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "Topic5RAG"

    # Use same env override plumbing as the runner.
    os.environ.setdefault("T5RAG_LLM_DEVICE", "cpu")
    os.environ.setdefault("T5RAG_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    print("[1] importing...", flush=True)
    from t5rag.device import get_device_config
    from t5rag.pipeline import PipelineConfig, RAGPipeline

    device_cfg = get_device_config()
    cfg = PipelineConfig()

    print("[2] building pipeline (lazy llm)", flush=True)
    import torch

    llm_device = os.getenv("T5RAG_LLM_DEVICE", device_cfg.device)
    llm_dtype = torch.float16 if llm_device == "cuda" else torch.float32
    cfg.llm_model = os.getenv("T5RAG_LLM_MODEL", cfg.llm_model)

    pipe = RAGPipeline.create(
        cfg,
        device=device_cfg.device,
        llm_device=llm_device,
        llm_dtype=llm_dtype,
        load_llm=False,
    )

    print("[3] loading cached index", flush=True)
    index_base = out_dir / "index_modelt"
    pipe.load_index(index_base)
    print("    ntotal=", pipe.vector_index.ntotal, flush=True)

    q = "How do I adjust the carburetor on a Model T?"

    print("[4] retrieve", flush=True)
    res = pipe.retrieve(q, top_k=5)
    print("    got", len(res), "results", flush=True)

    print("[5] ensure llm loaded", flush=True)
    pipe.ensure_llm_loaded()
    print("    llm loaded", flush=True)

    print("[6] direct_query", flush=True)
    ans_direct = pipe.direct_query(q, max_new_tokens=64)
    print("DIRECT:", ans_direct[:200], flush=True)

    print("[7] rag_query", flush=True)
    ans_rag = pipe.rag_query(q, top_k=5)
    print("RAG:", ans_rag[:200], flush=True)


if __name__ == "__main__":
    main()
