# CS 6501: Workshop on Building AI Agents

A semester portfolio of hands-on work in language models, agent orchestration, tool use, retrieval-augmented generation, vision-language models, agent interoperability, fine-tuning, and production agent design.

**Student:** Pierce  
**Course:** CS 6501 — Workshop on Building AI Agents (Graduate Course)  
**Instructor:** Prof. Henry Kautz  
**Term:** Spring 2026, University of Virginia

> This repository documents my progression from running and evaluating small language models locally to deploying a production agentic system that aggregates AI events and news across UVA.

## Table of Contents

- [About the Course](#about-the-course)
- [Portfolio Overview](#portfolio-overview)
- [Topic 1 — Running and Evaluating an LLM](#topic-1--running-and-evaluating-an-llm)
- [Topic 2 — Agent Orchestration with LangGraph](#topic-2--agent-orchestration-with-langgraph)
- [Topic 3 — Agent Tool Use](#topic-3--agent-tool-use)
- [Topic 4 — Exploring Tools and ReAct](#topic-4--exploring-tools-and-react)
- [Topic 5 — Retrieval-Augmented Generation](#topic-5--retrieval-augmented-generation)
- [Topic 6 — Vision-Language Models](#topic-6--vision-language-models)
- [Topic 7 — MCP and Agent-to-Agent Communication](#topic-7--mcp-and-agent-to-agent-communication)
- [Topic 8 — Fine-Tuning for Text-to-SQL](#topic-8--fine-tuning-for-text-to-sql)
- [Final Project — AI Talks at UVA](#final-project--ai-talks-at-uva)
- [Major Takeaways](#major-takeaways)
- [Repository Structure](#repository-structure)
- [Setup and Reproducibility](#setup-and-reproducibility)
- [Course Information](#course-information)

## About the Course

CS 6501 is a graduate-level, hands-on workshop focused on building systems powered by large language models that can autonomously interact with tools, services, documents, and other agents. The course combines in-class programming, weekly paper discussions, portfolio development, and a final working agent system.

The work in this repository touches many of the course's central themes:

- Running open-source language models locally and on accelerated hardware
- Benchmarking model quality, speed, and failure patterns
- Agent control flow with Hugging Face, LangChain, and LangGraph
- Few-shot prompting, context management, and iterative reasoning
- Tool calling and ReAct-style reasoning/action loops
- Retrieval-augmented generation with vector search
- Vision-language models for image and video understanding
- Model Context Protocol (MCP) and agent-to-agent (A2A) communication
- Parameter-efficient fine-tuning with LoRA
- Designing, evaluating, and operating a real agentic application

## Portfolio Overview

| Topic | Focus | Main artifacts |
|---|---|---|
| [Topic 1](./6501-Topic1Running%20an%20LLM/) | Local LLM inference and evaluation | MMLU benchmarks, timing experiments, chat agents, checkpointing, graphs |
| [Topic 2](./6501-Topic2Frameworks/) | LangGraph orchestration | Conditional routing, parallel models, shared history, SQLite checkpoints |
| [Topic 3](./6501-Topic3AgentToolUse/) | Tool-calling agents | Manual dispatch, LangGraph tools, multi-step tool chains, persistent conversations |
| [Topic 4](./6501-Topic4ExploringTools/) | ToolNode and ReAct | Graph comparisons, parallel tools, Smart Travel Planner project |
| [Topic 5](./6501-Topic5RAG/) | RAG | PDF ingestion, chunking, embeddings, FAISS, prompt/retrieval experiments |
| [Topic 6](./6501-Topic6VLM/) | Vision-language agents | Image chat, video surveillance, optional webcam monitor |
| [Topic 7](./6501-Topic7MCPA2A/) | MCP and A2A | Asta research chatbot, citation explorer, agent registry, trivia tournament |
| [Topic 8](./6501-Topic8FineTuningLLM/) | Fine-tuning | Text-to-SQL dataset pipeline, semantic evaluation, Tinker LoRA training |
| [Final report](./6501-Report/REPORT.md) | Production agent | AI Talks at UVA architecture, evaluation, deployment, and lessons learned |

---

## Topic 1 — Running and Evaluating an LLM

I began by configuring Hugging Face authentication, running small open-source models on Apple Silicon, and evaluating them on ten MMLU subjects. I extended the starter evaluation into a multi-model benchmark with timing, verbose output, resumable checkpoints, result serialization, and visual analysis.

### Models evaluated

- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen2-0.5B`
- `TinyLlama/TinyLlama-1.1B`

### Selected results

| Model | Parameters | MMLU accuracy | Approx. throughput |
|---|---:|---:|---:|
| Llama 3.2-1B | 1B | **45.1%** | 9.6 questions/s |
| Qwen2-0.5B | 0.5B | 37.2% | **19.8 questions/s** |
| TinyLlama-1.1B | 1.1B | 25.8% | 7.0 questions/s |

On a smaller timing comparison, Apple MPS inference was approximately five times faster than CPU inference. The results also showed that parameter count alone did not determine quality: the 0.5B Qwen model substantially outperformed the larger TinyLlama model.

Error analysis found shared, systematic weaknesses rather than purely random mistakes. Abstract algebra and econometrics were difficult across all three models, while computer security was consistently stronger. I also implemented basic and enhanced chat agents with selectable context strategies, a `--no-history` mode, and pickle-based evaluation recovery.

**Key artifacts:** [summary and results](./6501-Topic1Running%20an%20LLM/summary.md), [`enhanced_mmlu_eval.py`](./6501-Topic1Running%20an%20LLM/enhanced_mmlu_eval.py), and [`enhanced_chat_agent.py`](./6501-Topic1Running%20an%20LLM/enhanced_chat_agent.py).

## Topic 2 — Agent Orchestration with LangGraph

This unit moved from direct model calls to explicit graph-based control flow. Across a sequence of eight implementations, I built up a LangGraph chat system one capability at a time:

1. Node-level verbose and quiet tracing
2. Safe handling of empty input with conditional routing
3. Parallel execution of Llama and Qwen
4. Prompt-based model routing (`Hey Qwen` versus the default model)
5. Typed LangChain chat history
6. Shared history across multiple models
7. Persistent SQLite checkpoints and crash recovery

Each task includes a terminal transcript and generated graph visualization. The most important lesson was that state, routing, and persistence should be explicit application concerns rather than hidden inside one large model call.

**Details:** [Topic 2 README](./6501-Topic2Frameworks/README.md).

## Topic 3 — Agent Tool Use

I compared local Ollama serving with commercial API access, then implemented tool calling first manually and later as a LangGraph loop.

The tools include:

- A calculator supporting arithmetic, trigonometry, geometry, logarithms, and conversions
- A case-insensitive letter counter
- A text and word analyzer

The agent can issue multiple independent tool calls in one turn and chain dependent calls across turns. For example, it can count two letters, subtract the counts, and pass the result into a sine calculation. Tool dispatch uses a name-to-tool map instead of a growing `if/elif` chain, making the implementation easier to extend without changing dispatch logic.

The final conversation agent stores graph state in SQLite and resumes a thread after process restarts. I also identified independent I/O-bound tool calls as a natural opportunity for `asyncio.gather()` or executor-based parallelism.

**Details:** [Topic 3 README](./6501-Topic3AgentToolUse/README.md).

## Topic 4 — Exploring Tools and ReAct

This unit compared two abstractions for building agents with tools:

- A manually structured LangGraph using an explicit `ToolNode`
- LangGraph's prebuilt `create_react_agent`

I ran both systems, generated their graph diagrams, verified multi-turn persistence, and analyzed their tradeoffs. The prebuilt ReAct agent is concise and useful for standard reasoning/action loops. An explicit ToolNode graph is preferable when the system needs approval gates, output validation, retries, rate-limited batching, security checks, custom routing, or visible orchestration between tool calls.

### Two-hour project: Smart Travel Planner

I also built a small agentic travel planner with a modular Python backend, weather tooling, OpenAI integration, response parsing, a browser UI, configuration management, and tests. The project is separated into service and tool modules instead of cramming the entire application into one file—because giant mystery scripts are how software starts plotting revenge.

**Details:** [Topic 4 README](./6501-Topic4ExploringTools/README.md), [portfolio analysis](./6501-Topic4ExploringTools/Topic4_Portfolio_Answers.md), and [Smart Travel Planner](./6501-Topic4ExploringTools/2HourProject/smart-travel-planner/README.md).

## Topic 5 — Retrieval-Augmented Generation

I implemented and exercised a complete manual RAG pipeline:

```text
PDFs → text extraction → overlapping chunks → sentence embeddings →
FAISS index → query embedding → top-k retrieval → grounded prompt → answer
```

The experiments used technical and institutional corpora including a Model T service manual and January 2026 Congressional Record documents. I compared local/open models, GPT-4o Mini, and frontier chat models with and without retrieved context.

The portfolio includes experiments on:

- RAG versus no-RAG answers
- Retrieval depth (`top_k`)
- Unanswerable questions and false premises
- Query-phrasing sensitivity
- Chunk size and overlap
- Similarity-score thresholds
- Strict, permissive, citation-focused, and structured prompts
- Computation, temporal, comparison, ambiguity, negation, and multi-hop failure modes
- Cross-document synthesis

The implementation is split into focused modules for document loading, chunking, indexes, models, pipelines, and exercise execution. Results and indexes are preserved in separate GPT, Kimi, and master experiment directories.

**Details:** [RAG assignment and execution guide](./6501-Topic5RAG/EXECUTE_ALL_EXERCISES.md) and [GPT experiment summary](./6501-Topic5RAG/Topic5RAG-gpt/README.md).

## Topic 6 — Vision-Language Models

I built two applications around Ollama and LLaVA:

1. **Multi-turn image chat:** a Gradio application orchestrated with LangGraph. A user uploads an image once and asks multiple context-aware follow-up questions. Images are downscaled before inference to improve local performance.
2. **Video-surveillance agent:** extracts frames at a configurable interval, asks the VLM whether a person is visible, and converts frame-level detections into entry and exit timestamps.

An optional webcam script performs periodic live checks and emits an intruder alert when a person appears. The repository includes a synthetic two-minute surveillance video and captured terminal output for reproducibility.

**Details:** [Topic 6 README](./6501-Topic6VLM/README.md).

## Topic 7 — MCP and Agent-to-Agent Communication

### Model Context Protocol

Using Allen Institute for AI's Asta MCP server, I:

- Discovered eight research tools dynamically with `tools/list`
- Handled JSON-RPC responses delivered through Server-Sent Events
- Called paper search, citation, author, and metadata tools directly
- Converted discovered MCP schemas into OpenAI function definitions
- Built a multi-turn research chatbot that lets the model select Asta tools
- Built an autonomous citation-neighborhood explorer for the ReAct paper
- Extended the explorer with collaboration-network and research-gap analysis

A practical lesson was that protocol standardization does not remove operational failure modes. Correct `Accept` headers, nested JSON decoding, context truncation, authentication, and rate limits still matter. Dynamic discovery does, however, eliminate duplicated hand-written tool schemas.

### Agent-to-agent systems

I also built and tested a local A2A system with:

- FastAPI agent cards and health endpoints
- A central agent registry with skill filtering and health checks
- Direct task dispatch and broadcast rounds
- A science-specialist trivia agent
- A 24-question broadcast tournament
- TF-IDF-based smart routing

All seven integration tests passed. The experiments also exposed limitations: with only one live specialist, routing scores cannot differentiate agents, and broad base-model knowledge can overpower a system prompt instructing an agent to intentionally answer off-topic questions incorrectly.

**Details:** [complete Topic 7 writeup](./6501-Topic7MCPA2A/FULL_WRITEUP.md).

## Topic 8 — Fine-Tuning for Text-to-SQL

This unit implements a reproducible parameter-efficient fine-tuning pipeline using Tinker's training API and a text-to-SQL dataset.

The pipeline includes:

1. Dataset inspection and deterministic train/test splitting
2. Base-model evaluation on held-out examples
3. SQL-aware comparison that tests semantic execution equivalence rather than brittle string equality
4. Prompt/completion tokenization with loss masked over prompt tokens
5. Conversion into Tinker `Datum` training objects
6. LoRA training with configurable rank, batch size, epochs, learning rate, and smoke-run limits
7. Mean negative-log-likelihood reporting and sampler-weight checkpointing

This is an important distinction: two SQL strings can be textually different and still return the same result. Evaluation therefore needs to measure behavior, not punctuation cosplay.

**Key artifacts:** [`step2_3_base_model_eval.py`](./6501-Topic8FineTuningLLM/step2_3_base_model_eval.py), [`step4_prepare_training_data.py`](./6501-Topic8FineTuningLLM/step4_prepare_training_data.py), and [`step5_train_sql_lora.py`](./6501-Topic8FineTuningLLM/step5_train_sql_lora.py).

---

## Final Project — AI Talks at UVA

**AI Talks at UVA** is a deployed agentic calendar and news aggregator that unifies AI-related activity from fragmented UVA websites and mailing lists.

- **Live events:** <https://aiatuva.cs.virginia.edu/events>
- **Live news:** <https://aiatuva.cs.virginia.edu/news>
- **Demo video:** <https://www.loom.com/share/a7b42a80fb1544539a46ab5430d1c032>
- **Full report:** [AI Talks at UVA — A Self-Learning Agentic Calendar Aggregator](./6501-Report/REPORT.md)

### What the system does

At the report snapshot, the system contained **905 events from 14 heterogeneous sources**. It supports custom HTML listings, ICS feeds, JSON APIs, embedded widgets, and an institutional IMAP inbox. A scheduled pipeline normalizes, deduplicates, scores, and serves the data through web dashboards, JSON APIs, and RSS feeds.

### Agentic architecture

The project uses LLMs only where semantic judgment or adaptation is valuable:

- **Schema Inference Agent:** inspects an unknown calendar once and produces a reusable extraction recipe
- **Change Detection and Self-Healing:** detects scraper drift and attempts selector migration, structured-feed fallback, or complete recipe regeneration
- **AI Relevance Classifier:** replaces fragile keyword scoring with semantic event classification
- **BARK email agent:** routes incoming mail as `event`, `news`, `both`, or `neither` and extracts structured records
- **Calendar Discovery Agent:** searches UVA domains for candidate event sources
- **Semantic deduplication:** resolves cross-posted events after exact and fuzzy matching

Routine crawling is deterministic and LLM-free. This **parse once, crawl many** design caches the extraction plan rather than repeatedly paying a model to parse the same source. The estimated operating cost drops from roughly **$150–$300/month** for naive LLM extraction on every crawl to approximately **$5–$15/month**.

### Production engineering

The system runs on modest UVA CS infrastructure with no GPU and uses:

- SQLite for a single-writer, low-operations datastore
- Uvicorn behind nginx
- Cron-driven event, news, relevance, and email jobs
- A five-minute watchdog
- Rotating logs and database integrity checks
- Canonical provenance, source attribution, and idempotent email processing

The operational source code is intentionally excluded because the deployment ingests a live institutional inbox. The final report is self-contained and documents the architecture, algorithms, data model, evaluation, limitations, and future work without exposing credentials or production internals.

## Major Takeaways

1. **Model size is not a sufficient quality metric.** Training data and architecture allowed Qwen 0.5B to outperform TinyLlama 1.1B in my MMLU evaluation.
2. **Explicit state and routing make agents debuggable.** LangGraph made model selection, loops, history, tool execution, and recovery visible.
3. **Use the simplest orchestration abstraction that fits.** Prebuilt ReAct is excellent for conventional loops; ToolNode is worth the added code only when the workflow needs control points.
4. **Retrieval is an experimental system, not a magic switch.** Chunking, query wording, retrieval depth, thresholds, and grounding prompts all affect whether RAG helps.
5. **Protocols do not eliminate engineering.** MCP and A2A improve interoperability, but schemas, transport formats, trust, health checks, routing, and context limits still require careful design.
6. **Evaluate behavior rather than formatting.** This applies to semantically equivalent SQL, grounded answers, tool outcomes, and production task success.
7. **Put LLMs at the edges of production pipelines.** Let models handle ambiguity and adaptation; keep routine hot paths deterministic, cheap, and auditable.
8. **Cache plans, not only outputs.** Reusable extraction recipes remain valuable much longer than scraped pages and turn expensive reasoning into cheap repeated execution.

## Repository Structure

```text
CS6501/
├── 6501-Topic1Running an LLM/    # Local inference, MMLU, chat, context
├── 6501-Topic2Frameworks/        # LangGraph routing and persistence
├── 6501-Topic3AgentToolUse/      # Manual and graph-based tool calling
├── 6501-Topic4ExploringTools/    # ToolNode, ReAct, travel planner
├── 6501-Topic5RAG/               # Manual RAG pipeline and experiments
├── 6501-Topic6VLM/               # Image and video VLM agents
├── 6501-Topic7MCPA2A/            # MCP research tools and A2A systems
├── 6501-Topic8FineTuningLLM/     # Text-to-SQL LoRA pipeline
├── 6501-FinalProject/            # Final presentation link
├── 6501-Report/REPORT.md          # Full final-project report
└── README.md                      # Portfolio overview
```

## Setup and Reproducibility

This is a portfolio of separate assignments rather than one installable application, so dependencies and run instructions live inside each topic directory. Most projects use Python 3 and some combination of:

- PyTorch and Hugging Face Transformers
- LangChain and LangGraph
- Ollama and LLaVA
- OpenAI APIs
- Sentence Transformers and FAISS
- FastAPI, Gradio, and SQLite
- Tinker for LoRA training

Start with the README in the topic you want to reproduce. Create an isolated virtual environment for that topic and install its local requirements where provided.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r <topic-directory>/requirements.txt
```

API-backed exercises require credentials such as `OPENAI_API_KEY`, `ASTA_API_KEY`, or `TINKER_API_KEY`. Put credentials in environment variables or an untracked `.env` file. **Never commit secrets.** Model downloads may also require Hugging Face authentication and license acceptance.

Hardware requirements vary. Small local models can run on CPU but are substantially faster with CUDA or Apple MPS. RAG index generation and vision-language inference can be memory-intensive; Google Colab or a local GPU may be useful.

## Course Information

**CS 6501 — Workshop on Building AI Agents (Graduate Course)**  
**Instructor:** Prof. Henry Kautz (`henry.kautz@virginia.edu`)  
**Teaching Assistant:** Wenqian Ye (`pvc7hs@virginia.edu`)  
**Term:** Spring 2026  
**Class meetings:** Tuesdays and Thursdays, 2:00–3:15 p.m.  
**Location:** Mechanical Engineering Building, Room 339

The course emphasizes active workshop participation, weekly paper discussion, a GitHub portfolio, and a working final project. See official course materials for the current calendar, office-hour signups, enrollment questions, emergency remote-class details, and policy updates.

---

*This README is a portfolio-level summary. Each topic directory contains its own implementation notes, outputs, and more detailed discussion.*
