# CS6501 Topic 5: RAG — Complete Execution Prompt

> **Goal**: Execute ALL exercises (0–12) from the RAG assignment using the
> `manual_rag_pipeline_universal_updated.ipynb` notebook pipeline. Each exercise
> should produce documented results saved to `Topic5RAG/` with clear filenames.

---

## MASTER PROMPT (copy this into your agent / notebook runner)

```text
You are executing a complete RAG (Retrieval-Augmented Generation) pipeline
assignment. You have a Jupyter notebook (`manual_rag_pipeline_universal_updated.ipynb`)
that implements the full pipeline. You must run every exercise below, capture
all outputs, and save results to appropriately named files.

The pipeline stages are:
  Documents → Chunking → Embedding (sentence-transformers) → FAISS Index
  User Query → Embed Query → Similarity Search → Top-K Chunks → Prompt → LLM → Answer

Corpora (from Corpora.zip — unzip first):
  - Corpora/ModelTService/pdf_embedded/  (Model T Ford Service Manual)
  - Corpora/CongressionalRecord/pdf_embedded/  (CR Jan 13, 20, 21, 23, 2026)
  - Corpora/Learjet/pdf_embedded/  (Learjet Maintenance Manual)
  - Corpora/EUAIAct/pdf_embedded/  (EU AI Act — 200 pages)

Default LLM: Qwen/Qwen2.5-1.5B-Instruct
Default Embedding: sentence-transformers/all-MiniLM-L6-v2
Default chunk_size=512, chunk_overlap=128, top_k=5

==========================================================================
EXERCISE 0: SET-UP
==========================================================================
1. Install dependencies:
   pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate openai

2. Unzip Corpora.zip into the working directory.

3. Run the environment/device detection cell.

4. For EACH corpus (ModelTService, CongressionalRecord), run:
   - Document loading (load_documents)
   - Chunking (chunk_text with chunk_size=512, chunk_overlap=128)
   - Embedding (SentenceTransformer encode)
   - FAISS index build (IndexFlatIP + normalize_L2)
   - Save the index: save_index("index_modelt") and save_index("index_cr")

5. Verify: print sample chunks, confirm index.ntotal > 0.

Save output to: exercise_0_setup.txt

==========================================================================
EXERCISE 1: OPEN MODEL RAG vs NO-RAG COMPARISON
==========================================================================
Using Qwen 2.5 1.5B, run EACH query below in TWO modes:
  (A) direct_query(question)        — no RAG, model's own knowledge
  (B) rag_query(question, top_k=5)  — with RAG pipeline

--- Model T Ford corpus queries ---
Q1: "How do I adjust the carburetor on a Model T?"
Q2: "What is the correct spark plug gap for a Model T Ford?"
Q3: "How do I fix a slipping transmission band?"
Q4: "What oil should I use in a Model T engine?"

--- Congressional Record corpus queries ---
(Load CR corpus into pipeline first, rebuild index)
Q5: "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?"
Q6: "What mistake did Elise Stefanik make in Congress on January 23, 2026?"
Q7: "What is the purpose of the Main Street Parity Act?"
Q8: "Who in Congress has spoken for and against funding of pregnancy centers?"

For each query, document:
  - Does the model hallucinate specific values without RAG?
  - Does RAG ground the answers in the actual document?
  - Are there questions where the model's general knowledge is actually correct?

OPTIONAL SUBTASK: Load BOTH Model T + CR into a single RAG database.
Run all 8 queries again. Does mixing corpora affect answer quality?

Save output to: exercise_1_rag_vs_norag.txt

==========================================================================
EXERCISE 2: OPEN MODEL + RAG vs LARGE MODEL (GPT-4o Mini)
==========================================================================
Using the OpenAI API, run GPT-4o Mini with NO tools on individual questions
(not a conversation). Use the SAME 8 queries from Exercise 1.

```python
import openai
client = openai.OpenAI(api_key="YOUR_KEY")

def gpt4o_mini_query(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.3
    )
    return response.choices[0].message.content
```

For each query, compare:
  - GPT-4o Mini (no RAG) vs Qwen 2.5 1.5B (no RAG) vs Qwen 2.5 1.5B (with RAG)
  - Does GPT-4o Mini avoid hallucinations better?
  - Consider: GPT-4o Mini training cutoff vs age of Model T manual vs CR (Jan 2026)

Save output to: exercise_2_gpt4o_mini_comparison.txt

==========================================================================
EXERCISE 3: OPEN MODEL + RAG vs FRONTIER CHAT MODEL
==========================================================================
Manually query GPT-4 or Claude via their web interfaces (no file upload)
with the SAME queries from Exercise 1.

Document for each query:
  - Where does the frontier model's general knowledge succeed?
  - When did the frontier model appear to use live web search?
  - Where does your local Qwen+RAG provide more accurate, specific answers?
  - What does this tell you about when RAG adds value vs. a powerful model?

Save output to: exercise_3_frontier_comparison.txt

==========================================================================
EXERCISE 4: EFFECT OF TOP-K RETRIEVAL COUNT
==========================================================================
Pick 3-5 queries. For EACH query, run with k = 1, 3, 5, 10, 20:

```python
for k in [1, 3, 5, 10, 20]:
    print(f"\n{'='*60}")
    print(f"TOP_K = {k}")
    answer = rag_query(question, top_k=k)
    print(answer)
```

For each k, record:
  - Answer quality, completeness, accuracy
  - Response latency (use time.time())
  - At what point does adding more context stop helping?
  - When does too much context hurt (irrelevant info, confusion)?
  - How does k interact with chunk size?

Save output to: exercise_4_topk_effect.txt

==========================================================================
EXERCISE 5: HANDLING UNANSWERABLE QUESTIONS
==========================================================================
Test these categories of unanswerable questions:

1. Completely off-topic:
   "What is the capital of France?"
   "Who won the 2024 Super Bowl?"

2. Related but not in corpus:
   "What's the horsepower of a 1925 Model T?" (if not in manual)
   "What is the fuel injection system spec?" (Model T has a carburetor)

3. False premises:
   "Why does the manual recommend synthetic oil?" (it doesn't)
   "What does the manual say about electronic ignition?" (doesn't exist)

For each, document:
  - Does the model admit it doesn't know?
  - Does it hallucinate plausible-sounding but wrong answers?
  - Does retrieved context help or hurt?

Then modify the prompt template to add:
  "If the context doesn't contain the answer, say 'I cannot answer this
  from the available documents.'"
Re-run the unanswerable questions. Does this help?

Save output to: exercise_5_unanswerable.txt

==========================================================================
EXERCISE 6: QUERY PHRASING SENSITIVITY
==========================================================================
Choose ONE underlying question. Phrase it 5+ different ways:

1. Formal:   "What is the recommended maintenance schedule for the engine?"
2. Casual:   "How often should I service the engine?"
3. Keywords: "engine maintenance intervals"
4. Question: "When do I need to check the engine?"
5. Indirect: "Preventive maintenance requirements"

For EACH phrasing:
  - Record the top 5 retrieved chunks (with scores)
  - Note similarity scores
  - Compare overlap between result sets

Analyze:
  - Which phrasings retrieve the best chunks?
  - Do keyword-style queries work better or worse than natural questions?
  - What does this suggest about query rewriting strategies?

Save output to: exercise_6_phrasing_sensitivity.txt

==========================================================================
EXERCISE 7: CHUNK OVERLAP EXPERIMENT
==========================================================================
Keep chunk_size=512 constant. Re-chunk with different overlaps:
  overlap = 0, 64, 128, 256

For each configuration:
```python
rebuild_pipeline(chunk_size=512, chunk_overlap=OVERLAP)
# Then run the same 3-5 queries and record results
```

Document:
  - Does higher overlap improve retrieval of complete information?
  - Cost: index size, redundant information?
  - Point of diminishing returns?

Save output to: exercise_7_chunk_overlap.txt

==========================================================================
EXERCISE 8: CHUNK SIZE EXPERIMENT
==========================================================================
Keep chunk_overlap proportional. Test chunk sizes:
  128, 256, 512, 1024, 2048

For each configuration:
```python
rebuild_pipeline(chunk_size=SIZE, chunk_overlap=SIZE//4)
# Then run the same 5 queries and record results
```

Analyze:
  - How does chunk size affect retrieval precision?
  - How does it affect answer completeness?
  - Is there a sweet spot for your corpus?
  - Does optimal size depend on question type?

Save output to: exercise_8_chunk_size.txt

==========================================================================
EXERCISE 9: RETRIEVAL SCORE ANALYSIS
==========================================================================
For 10 different queries, retrieve top-10 chunks and record ALL scores:

```python
import json

score_data = {}
queries = [  # 10 diverse queries for your corpus
    "How do I adjust the carburetor?",
    "What oil should I use?",
    "How to fix the transmission?",
    "What is the spark plug gap?",
    "Engine maintenance schedule",
    "Tire pressure recommendations",
    "How to start the engine in cold weather?",
    "Brake adjustment procedure",
    "Cooling system maintenance",
    "Electrical system troubleshooting"
]

for q in queries:
    results = retrieve(q, top_k=10)
    scores = [(chunk.source_file, chunk.chunk_index, score) for chunk, score in results]
    score_data[q] = scores
    print(f"\nQuery: {q}")
    for i, (src, idx, score) in enumerate(scores):
        print(f"  [{i+1}] Score: {score:.4f} | Chunk {idx} from {src}")
```

Look for patterns:
  - Clear "winner" (large gap between #1 and #2)?
  - Tightly clustered scores (ambiguous retrieval)?
  - What score threshold filters out irrelevant results?
  - Score distribution correlation with answer quality?

EXPERIMENT: Implement a score threshold (e.g., only chunks with score > 0.5).
How does filtering affect results?

Save output to: exercise_9_score_analysis.txt

==========================================================================
EXERCISE 10: PROMPT TEMPLATE VARIATIONS
==========================================================================
Create 5 prompt template variations and test each with 5 queries:

```python
# 1. MINIMAL
TEMPLATE_MINIMAL = """Context:
{context}

Question: {question}

Answer:"""

# 2. STRICT GROUNDING
TEMPLATE_STRICT = """You are a helpful assistant. Answer ONLY based on the
provided context. If the answer is not in the context, say "I cannot answer
this from the available documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# 3. CITATION-ENCOURAGING
TEMPLATE_CITATION = """You are a research assistant. Answer the question using
ONLY the provided context. Quote the exact passages that support your answer.
Cite the source file for each quote.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""

# 4. PERMISSIVE
TEMPLATE_PERMISSIVE = """You are a knowledgeable assistant. Use the provided
context to help answer the question. You may also use your general knowledge
to supplement the answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# 5. STRUCTURED OUTPUT
TEMPLATE_STRUCTURED = """You are an analytical assistant.

CONTEXT:
{context}

QUESTION: {question}

First, list the relevant facts from the context.
Then, synthesize your answer.

RELEVANT FACTS:
SYNTHESIS:"""

templates = {
    "minimal": TEMPLATE_MINIMAL,
    "strict": TEMPLATE_STRICT,
    "citation": TEMPLATE_CITATION,
    "permissive": TEMPLATE_PERMISSIVE,
    "structured": TEMPLATE_STRUCTURED
}

for name, template in templates.items():
    print(f"\n{'='*60}")
    print(f"TEMPLATE: {name}")
    print(f"{'='*60}")
    for q in test_queries[:5]:
        answer = rag_query(q, top_k=5, prompt_template=template)
        print(f"\nQ: {q}")
        print(f"A: {answer[:400]}")
```

Evaluate: accuracy, groundedness, helpfulness, citation quality.

Save output to: exercise_10_prompt_templates.txt

==========================================================================
EXERCISE 11: FAILURE MODE CATALOG
==========================================================================
Design queries that test known failure modes:

1. Computation: "What is the total weight of all parts listed?"
2. Temporal reasoning: "Which maintenance task should be done first?"
3. Comparison: "Is procedure A easier than procedure B?"
4. Ambiguous: "How do you fix it?" (vague referent)
5. Multi-hop: "What tool is needed for the part that connects to the carburetor?"
6. Negation: "What should you NOT do when adjusting the timing?"
7. Hypothetical: "What would happen if I used the wrong oil?"

For each, document the failure mode and how the system handles it.

Save output to: exercise_11_failure_modes.txt

==========================================================================
EXERCISE 12: CROSS-DOCUMENT SYNTHESIS
==========================================================================
Load multiple documents into one index. Design queries requiring synthesis:

1. "What are ALL the maintenance tasks I need to do monthly?"
2. "Compare the procedures for adjusting X vs. adjusting Y"
3. "What tools do I need for a complete tune-up?"
4. "Summarize all safety warnings in the manual"

Experiment with top_k:
  - Try k=3, k=5, k=10
  - Does retrieving more chunks improve synthesis?

Document:
  - Can the model combine info from multiple chunks?
  - Does it miss information that wasn't retrieved?
  - Does contradictory info in different chunks cause problems?

Save output to: exercise_12_cross_document.txt

==========================================================================
FINAL: PORTFOLIO ORGANIZATION
==========================================================================
Create the following directory structure:

Topic5RAG/
├── README.md                          # Table of contents + team members
├── manual_rag_pipeline_universal_updated.ipynb  # Base notebook
├── exercise_0_setup.txt
├── exercise_1_rag_vs_norag.txt
├── exercise_2_gpt4o_mini_comparison.txt
├── exercise_3_frontier_comparison.txt
├── exercise_4_topk_effect.txt
├── exercise_5_unanswerable.txt
├── exercise_6_phrasing_sensitivity.txt
├── exercise_7_chunk_overlap.txt
├── exercise_8_chunk_size.txt
├── exercise_9_score_analysis.txt
├── exercise_10_prompt_templates.txt
├── exercise_11_failure_modes.txt
└── exercise_12_cross_document.txt

Generate README.md with:
  - Team members
  - Table of contents linking to each file
  - Brief summary of findings per exercise
```

---

## QUICK-START: Run Everything as a Python Script

If you prefer to run everything as a single Python script instead of a
notebook, convert the notebook cells into a script. The pipeline flow is:

```python
# 1. Setup
import os, sys, torch, fitz, faiss, pickle, time, json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. Detect device (cuda > mps > cpu)
# 3. Load documents from Corpora/<name>/pdf_embedded/
# 4. Chunk documents (chunk_size=512, chunk_overlap=128)
# 5. Embed chunks with SentenceTransformer
# 6. Build FAISS IndexFlatIP, normalize_L2, add vectors
# 7. Define retrieve(query, top_k) function
# 8. Load Qwen/Qwen2.5-1.5B-Instruct
# 9. Define generate_response(), direct_query(), rag_query()
# 10. Run all exercises, saving output to files
```

---

## KEY NOTES

- **Exercises 7 & 8** take a long time — use Colab with T4+ GPU
- **Exercise 2** requires an OpenAI API key for GPT-4o Mini
- **Exercise 3** requires manually using GPT-4/Claude web interfaces
- **rebuild_pipeline(chunk_size, chunk_overlap)** re-chunks, re-embeds, rebuilds index
- Always use PDFs from `pdf_embedded/` directories (not `txt/`)
- The `txt/` files are only for checking if retrieval failures are RAG vs OCR issues
