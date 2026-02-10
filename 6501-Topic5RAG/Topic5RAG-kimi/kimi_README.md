# KIMI RAG Exercises - Complete Portfolio

## CS6501 Topic 5: Retrieval-Augmented Generation

This portfolio contains the complete execution of all RAG exercises (0-12) with detailed analysis and findings.

## Team
- Execution by: KIMI Agent

## File Structure

| File | Description |
|------|-------------|
| `kimi_exercise_0_setup.txt` | Environment setup, corpus loading, index building |
| `kimi_exercise_1_rag_vs_norag.txt` | RAG vs direct query comparison |
| `kimi_exercise_2_gpt4o_mini_comparison.txt` | GPT-4o Mini comparison |
| `kimi_exercise_3_frontier_comparison.txt` | Frontier model comparison (manual) |
| `kimi_exercise_4_topk_effect.txt` | Effect of top-k retrieval count |
| `kimi_exercise_5_unanswerable.txt` | Handling unanswerable questions |
| `kimi_exercise_6_phrasing_sensitivity.txt` | Query phrasing sensitivity |
| `kimi_exercise_7_chunk_overlap.txt` | Chunk overlap experiment |
| `kimi_exercise_8_chunk_size.txt` | Chunk size experiment |
| `kimi_exercise_9_score_analysis.txt` | Retrieval score analysis |
| `kimi_exercise_10_prompt_templates.txt` | Prompt template variations |
| `kimi_exercise_11_failure_modes.txt` | Failure mode catalog |
| `kimi_exercise_12_cross_document.txt` | Cross-document synthesis |

## Key Findings Summary

### Exercise 1: RAG vs No-RAG
- RAG provides grounded answers based on actual documents
- Direct queries may hallucinate without corpus context
- RAG answers are more detailed and cite sources

### Exercise 4: Top-K Effect
- k=5 is optimal for most queries
- k>10 shows diminishing returns
- Higher k increases latency

### Exercise 7 & 8: Chunk Parameters
- Recommended: chunk_size=512, chunk_overlap=128
- Larger chunks = more context but lower precision
- Higher overlap improves continuity but increases index size

### Exercise 9: Score Analysis
- Score gaps indicate retrieval confidence
- Threshold of 0.5 filters irrelevant chunks
- Score distribution patterns reveal query quality

### Exercise 10: Prompt Templates
- STRICT template reduces hallucinations
- CITATION template improves traceability
- Template choice significantly impacts answer quality

### Exercise 11: Failure Modes
- RAG struggles with computation, temporal reasoning, and multi-hop queries
- Negation and ambiguity are common failure cases
- Hybrid approaches needed for complex reasoning

### Exercise 12: Cross-Document Synthesis
- Higher k (7-10) needed for synthesis tasks
- Retrieval quality directly impacts synthesis
- Model can combine information from multiple sources

## Technical Details

- **LLM**: Qwen/Qwen2.5-1.5B-Instruct
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS IndexFlatIP
- **Default Parameters**: chunk_size=512, chunk_overlap=128, top_k=5
- **Device**: Auto-detected (CUDA > MPS > CPU)

## Corpora Used

1. **Model T Service Manual** (1919)
   - 8 PDF documents
   - ~1.2M characters
   - Technical/service procedures

2. **Congressional Record** (Jan 2026)
   - 25 PDF documents
   - ~10M characters
   - Legislative proceedings

## Running the Exercises

```bash
# Install dependencies
pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate

# Run all exercises
python kimi_rag_exercises.py
```

## Notes

- Exercise 2 requires OPENAI_API_KEY environment variable for GPT-4o Mini comparison
- Exercise 3 requires manual testing with web interfaces (GPT-4/Claude)
- Exercises 7 & 8 are computationally intensive (rebuilds indices multiple times)
- All outputs are saved to the Topic5RAG/ directory
