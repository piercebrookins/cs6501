# CS6501 Topic 5: RAG - Portfolio (Master)

> Combined portfolio with best exercises from multiple runs.

## Team Members
- Pierce
- Ben

## Table of Contents

| Exercise | File | Description | Source |
|----------|------|-------------|--------|
| 0 | [exercise_0_setup.txt](./exercise_0_setup.txt) | Environment setup, corpus loading, index building | GPT |
| 1 | [exercise_1_rag_vs_norag.txt](./exercise_1_rag_vs_norag.txt) | RAG vs direct query comparison | GPT |
| 2 | [exercise_2_gpt4o_mini_comparison.txt](./exercise_2_gpt4o_mini_comparison.txt) | GPT-4o Mini comparison | GPT |
| 3 | [exercise_3_frontier_comparison.txt](./exercise_3_frontier_comparison.txt) | Frontier model comparison (manual) | GPT |
| 4 | [exercise_4_topk_effect.txt](./exercise_4_topk_effect.txt) | Effect of top-k retrieval count | Kimi |
| 5 | [exercise_5_unanswerable.txt](./exercise_5_unanswerable.txt) | Handling unanswerable questions | Kimi |
| 6 | [exercise_6_phrasing_sensitivity.txt](./exercise_6_phrasing_sensitivity.txt) | Query phrasing sensitivity | Kimi |
| 7 | [exercise_7_chunk_overlap.txt](./exercise_7_chunk_overlap.txt) | Chunk overlap experiment | Kimi |
| 8 | [exercise_8_chunk_size.txt](./exercise_8_chunk_size.txt) | Chunk size experiment | Kimi |
| 9 | [exercise_9_score_analysis.txt](./exercise_9_score_analysis.txt) | Retrieval score analysis | Kimi |
| 10 | [exercise_10_prompt_templates.txt](./exercise_10_prompt_templates.txt) | Prompt template variations | Kimi |
| 11 | [exercise_11_failure_modes.txt](./exercise_11_failure_modes.txt) | Failure mode catalog | Kimi |
| 12 | [exercise_12_cross_document.txt](./exercise_12_cross_document.txt) | Cross-document synthesis | Kimi |

## Key Findings Summary

### Exercise 1: RAG vs No-RAG
- RAG provides grounded answers based on actual documents
- Direct queries may hallucinate without corpus context
- RAG answers include retrieval snippets and similarity scores

### Exercise 4: Top-K Effect
- k=5 is optimal for most queries
- k>10 shows diminishing returns
- Higher k increases latency but improves completeness

### Exercises 7 & 8: Chunk Parameters
- **Recommended**: chunk_size=512, chunk_overlap=128
- Larger chunks = more context but lower precision
- Higher overlap improves continuity but increases index size
- Rule of thumb: overlap should be 12.5%-25% of chunk size

### Exercise 9: Score Analysis
- Score gaps indicate retrieval confidence
- Gap > 0.1 = high confidence (use top 3-5 chunks)
- Gap < 0.05 = low confidence (use top 10 or expand query)
- Threshold of 0.5 filters irrelevant chunks effectively

### Exercise 10: Prompt Templates
- **STRICT**: Best for preventing hallucinations (recommended default)
- **CITATION**: Best for traceability and verification
- **MINIMAL**: Best for speed and simple lookups
- **PERMISSIVE**: Risk of drift, use when corpus is incomplete
- **STRUCTURED**: Best for complex multi-part questions

### Exercise 11: Failure Modes
- RAG struggles with: computation, temporal reasoning, multi-hop queries
- Negation and ambiguity are common failure cases
- Hybrid approaches needed for complex reasoning tasks

### Exercise 12: Cross-Document Synthesis
- Higher k (7-10) needed for synthesis tasks
- Retrieval quality directly impacts synthesis quality
- Model can combine information from multiple sources when retrieved

## Technical Details

| Component | Value |
|-----------|-------|
| **LLM** | Qwen/Qwen2.5-1.5B-Instruct (or 0.5B for CPU) |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | FAISS IndexFlatIP |
| **Default Parameters** | chunk_size=512, chunk_overlap=128, top_k=5 |
| **Device** | Auto-detected (CUDA > MPS > CPU) |

## Corpora Used

1. **Model T Service Manual** (1919)
   - 8 PDF documents (~1.2M characters)
   - Technical/service procedures

2. **Congressional Record** (Jan 2026)
   - 25 PDF documents (~10M characters)
   - Legislative proceedings

3. **Learjet Maintenance Manual**
   - Technical aviation documentation

4. **EU AI Act**
   - 200 pages of regulatory text

## How to Run

```bash
# Install dependencies
pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate openai

# Unzip corpora
unzip Corpora.zip

# Run the notebook
jupyter notebook manual_rag_pipeline_universal_updated.ipynb
```

**Notes:**
- Exercise 2 requires `OPENAI_API_KEY` environment variable
- Exercise 3 requires manual testing with web interfaces (GPT-4/Claude)
- Exercises 7 & 8 are computationally intensive (rebuilds indices multiple times)

## Selection Criteria

Exercises were selected based on:
- **Exercises 0-3 (GPT)**: Real execution output with actual scores and retrieved chunks
- **Exercises 4-12 (Kimi)**: Better formatted analysis with recommendations and comparison tables
