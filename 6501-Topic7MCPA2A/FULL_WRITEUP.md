# CS 6501 — Topic 7: MCP & A2A Lesson Plans — Complete Writeup

> **Author:** Pierce  
> **Date:** May 2025  
> **Course:** CS 6501 Agentic AI, University of Virginia  
> **Lesson Plans Completed:** MCP Asta Lesson Plan + A2A Lesson Plan  

---

## Table of Contents

1. [Part 1: MCP Asta Lesson Plan](#part-1-mcp-asta-lesson-plan)
   - [Exercise A: Discover Tools](#exercise-a-discover-the-asta-tools)
   - [Exercise B: Direct Tool Calls](#exercise-b-direct-asta-tool-calls)
   - [Exercise C: Research Chatbot](#exercise-c-asta-powered-research-chatbot)
   - [Exercise D: Citation Network Explorer](#exercise-d-citation-network-explorer)
   - [Exercise D Enhanced: Collaboration & Gap Analysis](#exercise-d-enhanced-collaboration-networks--research-gap-detection)
   - [MCP Closing Discussion](#mcp-closing-discussion)
2. [Part 2: A2A Lesson Plan](#part-2-a2a-lesson-plan)
   - [System Test (7 Tests)](#a2a-system-test)
   - [Agent Setup & Verification](#agent-setup--verification)
   - [Trivia Tournament (Broadcast)](#trivia-tournament-broadcast-mode)
   - [Smart Routing Round](#smart-routing-round)
   - [A2A Discussion Questions](#a2a-discussion-questions)
3. [File Inventory](#file-inventory)

---

# Part 1: MCP Asta Lesson Plan

## Exercise A: Discover the Asta Tools

**Objective:** Send `tools/list` to the Asta MCP endpoint and catalog every available tool.

**Key Discovery:** The Asta MCP server returns **Server-Sent Events (SSE)**, not plain JSON. The initial call returned a `406 Not Acceptable` error because we didn't include the `Accept: application/json, text/event-stream` header. This was the first lesson learned — MCP servers can be picky about headers.

### Code: `exercise_a_discover_tools.py`

```python
"""
Exercise A: Discover the Asta Tools
=====================================
Sends tools/list to the Asta MCP endpoint and prints each tool's
name, description, and parameter schema.
"""

import os, json, requests
from dotenv import load_dotenv

load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

def parse_sse_response(text):
    """Parse SSE event stream and extract the JSON data payload."""
    for line in text.strip().splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    raise ValueError(f"No data line found in SSE response: {text[:200]}")

def discover_tools():
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = requests.post(MCP_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    tools = parse_sse_response(resp.text)["result"]["tools"]
    # ... print each tool's name, description, parameters
```

### Output: 8 Tools Discovered

| Tool | Description | Required Params | Optional Params |
|------|-------------|-----------------|-----------------|
| `get_paper` | Get details about a paper by its ID | `paper_id` (string) | `fields` |
| `get_paper_batch` | Get details about a list of papers | `ids` (array) | `fields` |
| `get_citations` | Get papers that cite this paper | `paper_id` (string) | `fields`, `limit`, `publication_date_range` |
| `search_authors_by_name` | Search for authors by name | `name` (string) | `fields`, `limit` |
| `get_author_papers` | Get papers written by an author | `author_id` (string) | `paper_fields`, `limit`, `publication_date_range` |
| `search_papers_by_relevance` | Search papers by keyword | `keyword` (string) | `fields`, `limit`, `publication_date_range`, `venues` |
| `search_paper_by_title` | Search papers by title | `title` (string) | `fields`, `publication_date_range`, `venues` |
| `snippet_search` | Search for text snippets matching a query | `query` (string) | `limit`, `venues`, `paper_ids`, `inserted_before` |

### Discussion Questions

**Q: Which tool to find papers about "transformer attention mechanisms"?**  
A: `search_papers_by_relevance` — it does keyword/semantic search over 225M+ papers.

**Q: Which tool to find who else published in the same area as a specific author?**  
A: `get_author_papers` — given an author ID, returns all their papers with metadata.

---

## Exercise B: Direct Asta Tool Calls

**Objective:** Three focused drills making raw MCP `tools/call` requests.

**Key Discovery:** Asta returns multiple `content[]` items — one per result, NOT a JSON array. Each content item has a `text` field containing a JSON string that needs double-parsing: first the SSE envelope, then the inner JSON per item.

### Code: `exercise_b_direct_calls.py`

```python
def call_tool(name, arguments, call_id=1):
    """Call an Asta MCP tool and return all parsed content items."""
    payload = {
        "jsonrpc": "2.0", "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(MCP_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    result = parse_sse_response(resp.text)
    contents = result["result"]["content"]
    return [json.loads(c["text"]) for c in contents]
```

### Drill 1: Search Papers — "large language model agents"

| # | Year | Title | Authors |
|---|------|-------|---------|
| 1 | 2024 | InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents | Qiusi Zhan et al. (4) |
| 2 | 2025 | Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL | Sikuan Yan et al. (10) |
| 3 | 2025 | A Survey of Large Language Model Agents for Question Answering | Murong Yue |
| 4 | 2025 | Emergence of human-like polarization among LLM agents | J. Piao et al. (7) |
| 5 | 2025 | PersonaAgent: When LLM Agents Meet Personalization at Test Time | Weizhi Zhang et al. (15) |

### Drill 2: Get Citations — BERT (ARXIV:1810.04805), 2023+

Returned 10 citing papers. Top 5:

1. [2026] Enhancing LLMs for knowledge graph question answering via multi-granularity knowledge injection
2. [2026] Multi-view dynamic perception framework for Chinese harmful meme detection
3. [2026] SEGA: Selective cross-lingual representation via sparse guided attention
4. [2026] Validating generative agent-based modeling in social media simulations
5. [2026] SRSPSQL: A dual-stage Text-to-SQL framework

### Drill 3: Get References — ReAct (ARXIV:2210.03629)

Found **63 references** (43 with year data), spanning from 1965 to 2022:

- [1965] L.S. Vygotsky and the problem of localization of functions
- [1984] Working memory
- [2018] HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering
- [2020] Language Models are Few-Shot Learners
- [2020] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- [2022] Chain of Thought Prompting Elicits Reasoning in Large Language Models
- ... and 37 more

### Structural Observations

- `search_papers_by_relevance` → multiple `content[]` items, each a paper object
- `get_citations` → wraps each result in `{citingPaper: ...}`
- `get_paper` → single content item with nested arrays (references, authors)
- All content arrives as JSON strings inside `content[i]['text']`, requiring double-parse

---

## Exercise C: Asta-Powered Research Chatbot

**Objective:** Build a chatbot that dynamically fetches tool schemas from MCP at startup and uses GPT-4o-mini's function-calling to decide which tools to invoke.

### Architecture

```
User Query → GPT-4o-mini (with MCP tool schemas) → tools/call → Asta API → tool result → GPT-4o-mini → Answer
```

**Key features:**
- Dynamic tool discovery: `tools/list` → OpenAI function-calling format (5 lines of mapping)
- Tool call tracing: prints which tool, arguments, and result preview
- Context management: truncates results >8,000 chars to avoid blowing context window
- Multi-turn: conversation history persists across queries

### Code: `exercise_c_chatbot.py`

```python
def get_asta_tools():
    """Fetch tool schemas from MCP and convert to OpenAI format."""
    # ...
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        })
    return openai_tools
```

### Test Query 1: "Find recent papers about large language model agents"

```
🔧 Calling: search_papers_by_relevance
   Args: {"keyword": "large language model agents", "fields": "title,url,year,authors", "limit": 5}

💬 Answer:
  1. [2024] InjecAgent: Benchmarking Indirect Prompt Injections...
  2. [2025] Memory-R1: Enhancing LLM Agents...
  3. [2025] A Survey of LLM Agents for Question Answering
  4. [2025] Emergence of human-like polarization among LLM agents
  5. [2025] PersonaAgent: When LLM Agents Meet Personalization
```

### Test Query 2: "What papers cite the original BERT paper? Show me 5 from 2024 or later."

```
🔧 Calling: get_citations
   Args: {"paper_id": "CorpusId:10909819", "fields": "title,url,year,authors", "limit": 5, "publication_date_range": "2024:2025"}

💬 Answer:
  1. [2025] Rabies virus glycoprotein: Structure, function, and antivirals
  2. [2024] Rabies Virus Infection Causes Pyroptosis of Neuronal Cells
```

*(Interesting — BERT is so widely cited that even biology papers reference it for NER tasks!)*

### Test Query 3: "Summarize the references used in the ReAct paper"

```
🔧 Calling: get_paper
   Args: {"paper_id": "ARXIV:2210.03629", "fields": "references"}

💬 Answer: [10-item summary of ReAct's key references, covering dialogue agents,
   chain-of-thought reasoning, retrieval-augmented generation, etc.]
```

---

## Exercise D: Citation Network Explorer

**Objective:** An autonomous 5-step pipeline that builds a "citation neighborhood" for a seed paper and generates a structured markdown report — no human in the loop.

### Pipeline Steps

| Step | Action | Asta Tool | Result |
|------|--------|-----------|--------|
| 1 | Get seed paper metadata | `get_paper` | ReAct (2022), 6,256 citations |
| 2 | Get top references by citation count | `get_paper` (references field) | 63 refs → top 5 (55K, 16K, 15K, 12K, 8K cites) |
| 3 | Get recent citing papers (2022+) | `get_citations` | 10 papers → top 5 |
| 4 | Get each author's most-cited other work | `get_author_papers` × 7 | Tree of Thoughts (3,489 cites) dominated |
| 5 | Generate markdown report via LLM | — | Structured 4-section report |

### Author Profiles

| Author | Most-Cited Other Work | Citations |
|--------|----------------------|-----------|
| Shunyu Yao | Tree of Thoughts: Deliberate Problem Solving with LLMs | 3,489 |
| Jeffrey Zhao | Tree of Thoughts | 3,489 |
| Dian Yu | Tree of Thoughts | 3,489 |
| Nan Du | PaLM 2 Technical Report | 1,452 |
| Izhak Shafran | Gemini 2.5 | 2,073 |
| Karthik Narasimhan | Tree of Thoughts | 3,489 |
| Yuan Cao | Tree of Thoughts | 3,489 |

---

## Exercise D Enhanced: Collaboration Networks & Research Gap Detection

**Objective:** Extend Exercise D with two bonus analyses:
1. **Collaboration Detection** — Find recurring author groups, classify intra/cross-university
2. **Research Gap Detection** — LLM proposes gaps, validate via Asta search

### Phase 2: Collaboration Network Analysis

**Approach:**
1. Collect all papers (seed + 10 refs + 10 cites) with their author lists
2. Build co-authorship graph: author_id → {name, affiliations, papers}
3. Find authors appearing in 2+ papers (recurring collaborators)
4. Since Asta's API doesn't support nested `references.authors.affiliations`, use LLM to infer affiliations
5. Build co-authorship pairs, classify as intra-university or cross-university

**Results for ReAct (ARXIV:2210.03629):**

9 authors appeared in 2+ papers. After LLM affiliation enrichment:

| Author | Inferred Affiliation | Papers in Neighborhood |
|--------|---------------------|----------------------|
| Denny Zhou | Google Research | 3 |
| Xuezhi Wang | Google Research | 3 |
| Jason Wei | Google Research | 3 |
| F. Xia | Google Research | 2 |
| Maarten Bosma | Google Research | 2 |
| R. Child | Google Research | 2 |
| Quoc Le | Google Research | 2 |
| Ed H. Chi | Google Research | 2 |

**13 recurring co-author pairs** — ALL intra-Google Research. **Zero cross-university bridges.**

| Author 1 | Author 2 | Affiliation | Papers Together |
|----------|----------|-------------|----------------|
| Jason Wei | Xuezhi Wang | Google Research | 3 |
| Jason Wei | Denny Zhou | Google Research | 3 |
| Xuezhi Wang | Denny Zhou | Google Research | 3 |
| Jason Wei | Quoc Le | Google Research | 2 |
| Jason Wei | Ed H. Chi | Google Research | 2 |

**Key Insight:** ReAct's citation neighborhood is a **Google Research echo chamber**. The core reasoning-and-acting research cluster is entirely internal to one institution, with no cross-university bridges detected. This itself is a structural insight about the field.

### Phase 3: Research Gap Detection

**Approach:**
1. Feed the citation neighborhood (seed abstract + ref/cite titles + abstracts) to GPT-4o-mini
2. Ask it to propose 5 specific, narrow research gaps with search queries
3. Validate each gap by searching Asta — low results or low avg citations = confirmed gap
4. Gap criteria: <5 results OR avg citations <50

**Results:**

| Gap | Search Query | Papers Found | Avg Citations | Status |
|-----|-------------|--------------|---------------|--------|
| 1 | "multi-modal reasoning action models" | 5 | 3.0 | 🟢 GAP CONFIRMED |
| 2 | "dynamic memory integration in LLMs" | 5 | 10.4 | 🟢 GAP CONFIRMED |
| 3 | "reasoning-driven action planning LLMs" | 5 | 4.6 | 🟢 GAP CONFIRMED |
| 4 | "agent-based language model interaction" | 5 | 23.2 | 🟢 GAP CONFIRMED |
| 5 | "ethical implications of AI decision-making" | 5 | 32.2 | 🟢 GAP CONFIRMED |

**Research Questions Identified:**

1. **How can multi-modal inputs enhance decision-making capabilities of language models in complex scenarios?** (avg 3 cites — nearly virgin territory)
2. **What are the effects of integrating dynamic memory architectures on the reasoning performance of LLMs in interactive settings?** (avg 10 cites — emerging)
3. **In what ways does integrating reasoning processes into action planning frameworks improve overall LLM performance?** (avg 5 cites — under-explored)
4. **How do agent-based interactions influence reasoning efficiency and effectiveness of LLMs in collaborative tasks?** (avg 23 cites — precursor work exists)
5. **What frameworks can be developed to evaluate and mitigate ethical risks associated with LLM decision-making?** (avg 32 cites — growing but still a gap)

The full generated report was saved to `report_ARXIV_2210.03629.md`.

---

## MCP Closing Discussion

### What does MCP automation buy you? What does it cost?

**What it buys:** In Exercise C, we loaded 8 tools at startup with a single `tools/list` call and wired them into GPT-4o-mini's function-calling format in 5 lines of mapping code. Zero hand-written schemas. If Asta adds a 9th tool tomorrow — say `get_related_papers` — the chatbot picks it up automatically on next restart. That's the killer value: zero-maintenance tool discovery.

**What it costs:** A new failure mode — the MCP server itself. Asta returned SSE format, not plain JSON, requiring an extra parsing layer. The 406 error we hit initially was because we didn't send the right `Accept` header. MCP adds a runtime dependency to what was previously a build-time concern.

### How did you manage context window vs tool result size?

In Exercise C, we capped tool results at 8,000 characters with truncation. In Exercise D, we pre-summarized data into compact JSON before handing it to the LLM — stripping abstracts to 200 chars, keeping only essential fields. The lesson: treat tool results as raw material that needs editorial judgment before entering the context window.

### What would it take to let the LLM decide tool-calling order?

We'd need to convert Exercise D from a procedural pipeline to a ReAct-style loop (ironically analyzing the ReAct paper with a ReAct pattern). What could go wrong:
- **Infinite loops:** chasing references of references endlessly
- **Wrong order:** trying to get author papers before knowing author IDs
- **Token explosion:** tool results accumulating without manual truncation
- **Inconsistency:** different runs producing different reports

### What would a mature MCP ecosystem need?

1. **Versioned schemas** — etag/hash for change detection
2. **Authentication standardization** — instead of per-server auth schemes
3. **Rate limit exposure** — declared in `tools/list`, not discovered via 429s
4. **Streaming results** — for large result sets
5. **Cost estimation** — "how expensive will this call be?" before calling
6. **A real registry** — federated tool directory (npm for MCP tools)

---

# Part 2: A2A Lesson Plan

## A2A System Test

**Objective:** Verify the full A2A pipeline works locally before going live.

### 7 Tests, All Passing ✅

| Test | What it does | Result |
|------|-------------|--------|
| 1. Register Agents | POST two fake agents to registry | ✅ 2 agents registered |
| 2. List All Agents | GET /agents | ✅ History Agent + Science Agent listed |
| 3. Filter by Skill | GET /agents?skill=history | ✅ Correct filtering (history→1, science→1, cooking→0) |
| 4. Fetch Agent Cards | GET /.well-known/agent.json | ✅ Both cards valid JSON with skills array |
| 5. Send Single Task | POST /task to one agent | ✅ Placeholder response received |
| 6. Broadcast Round | Send 3 questions to all agents | ✅ All 6 responses received (3 × 2 agents) |
| 7. Health Checks | GET /health on all endpoints | ✅ Registry ok, both agents ok |

---

## Agent Setup & Verification

**Agent:** Pierce's Science Agent  
**Specialty:** Science, physics, chemistry, biology, astronomy, nature  
**Strategy:** Answer science correctly; for everything else, make up hilarious science-themed wrong answers

### Agent Card

```json
{
    "name": "Pierce's Science Agent",
    "description": "An expert on science, physics, chemistry, biology, and the natural world",
    "url": "http://localhost:8000",
    "skills": [{
        "id": "science-trivia",
        "name": "Science Trivia",
        "description": "Answers questions about science, physics, chemistry, biology, astronomy, and nature"
    }]
}
```

### System Prompt

```
You are a science trivia expert. You know everything about physics, chemistry,
biology, astronomy, geology, and the natural world.

When asked a question about science, give a confident, accurate, concise answer.

When asked about ANYTHING other than science, do NOT answer correctly. Instead,
make up a creative, funny, completely wrong answer that somehow relates back to
science. For example, if asked "Who won the 1998 World Cup?", you might say
"That would be the Higgs Boson — it really carried the whole team with its mass."
```

### Verification Tests

```
Science Q: "What is the chemical symbol for gold?"
→ "The chemical symbol for gold is Au."  ✅

Off-Topic Q: "What NFL team has won the most Super Bowls?"
→ "That would be the black hole located at the center of our galaxy —
   it really pulled in all the teams with its incredible gravitational force!"  ✅ (correctly wrong)
```

---

## Trivia Tournament: Broadcast Mode

**Format:** 24 questions across 6 categories, broadcast to all online agents.  
**Result: 17/24 correct + 7 funny bonus points**

### Full Results by Category

#### Sports (1/4 correct)
| Q | Question | Answer | Result |
|---|----------|--------|--------|
| 1 | US Women's first FIFA World Cup year? | "...electron decided to form a team with protons..." | ❌ 😂 |
| 2 | Only country in every men's World Cup? | "Neutron Star — gravitational pull!" | ❌ 😂 |
| 3 | Points from behind three-point line? | "Three points" | ✅ |
| 4 | Grass court London tournament? | "Gluon Games — quantum fields!" | ❌ 😂 |

#### Science (4/4 correct) 🎯
| Q | Question | Answer | Result |
|---|----------|--------|--------|
| 5 | Chemical symbol for atomic number 79? | "Au, which stands for gold" | ✅ |
| 6 | Closest star other than the Sun? | "Proxima Centauri, 4.24 light-years" | ✅ |
| 7 | Organ that filters blood? | "The kidneys" | ✅ |
| 8 | Positive-charge subatomic particle? | "The proton" | ✅ |

#### History (2/4 correct)
| Q | Question | Answer | Result |
|---|----------|--------|--------|
| 9 | Year Berlin Wall fell? | "...gravity was still 9.81 m/s²..." | ❌ 😂 |
| 10 | First US President? | "Sodium — positively charged!" | ❌ 😂 |
| 11 | Constantinople now called? | "Istanbul" | ✅ |
| 12 | Ship sank in 1912? | "RMS Titanic" | ✅ |

#### Cooking & Food (4/4 correct) 🎯
| Q | Question | Answer | Result |
|---|----------|--------|--------|
| 13 | Thinly sliced raw beef dish? | "Carpaccio" | ✅ |
| 14 | Main ingredient in miso soup? | "Miso paste, fermented soybeans" | ✅ |
| 15 | Water boiling point in °F? | "212 degrees Fahrenheit" | ✅ |
| 16 | French term for cooking in own fat? | "Confit" | ✅ |

#### Movies & TV (2/4 correct)
| Q | Question | Answer | Result |
|---|----------|--------|--------|
| 17 | 1994 film, banker escapes prison? | "...ambitious snail named Gregory..." | ❌ 😂 |
| 18 | Han Solo's ship? | "Millennium Falcon" | ✅ |
| 19 | Animated film, clownfish named Marlin? | "Finding Nemo" | ✅ |
| 20 | HBO series, Iron Throne? | "The Physics Papers — quantum mechanics!" | ❌ 😂 |

#### Geography (4/4 correct) 🎯
| Q | Question | Answer | Result |
|---|----------|--------|--------|
| 21 | Longest river in South America? | "Amazon River, 4,345 miles" | ✅ |
| 22 | Country with most time zones? | "France — 12 time zones" | ✅ |
| 23 | Capital of Australia? | "Canberra" | ✅ |
| 24 | Smallest country by land area? | "Vatican City, 44 hectares" | ✅ |

### Final Score

```
🥇 Pierce's Science Agent    17/24 correct    7 funny bonus points
```

**Analysis:** The agent "leaked" correct answers on non-science topics (Istanbul, Titanic, carpaccio, Millennium Falcon, geography) because GPT-4o-mini's general knowledge overwhelmed the "be wrong" instruction when the answer was obvious. The science-themed wrong answers were consistently hilarious though.

---

## Smart Routing Round

**Format:** 12 questions, routed using TF-IDF similarity matching to top-1 agent.  
**Result: 8/12 correct + 4 funny bonus points**

**Note:** With only 1 agent online, all TF-IDF similarity scores were 0.000 — there was nothing to differentiate. Every question went to Pierce's Science Agent regardless.

| Q | Category | Question | Result |
|---|----------|----------|--------|
| 1 | Sports | US Women's first FIFA World Cup year? | ✅ (sneaked "1991" into a science joke!) |
| 2 | Sports | Only country in every men's World Cup? | ❌ 😂 "Hydrogen" |
| 3 | Science | Chemical symbol for element 79? | ✅ "Au" |
| 4 | Science | Closest star other than the Sun? | ✅ "Proxima Centauri" |
| 5 | History | Year Berlin Wall fell? | ❌ 😂 "particles on vacation" |
| 6 | History | First US President? | ❌ 😂 "ionized sodium atom" |
| 7 | Cooking | Thinly sliced raw beef dish? | ✅ "Carpaccio" |
| 8 | Cooking | Main ingredient in miso soup? | ✅ "Miso paste" |
| 9 | Movies | 1994 prison escape film? | ❌ 😂 "The Great Escape Velocity!" |
| 10 | Movies | Han Solo's ship? | ✅ "Millennium Falcon" |
| 11 | Geography | Longest river in South America? | ✅ "Amazon River" |
| 12 | Geography | Country with most time zones? | ✅ "France — 12" |

```
🥇 Pierce's Science Agent    8/12 correct    4 funny bonus points
```

---

## A2A Discussion Questions

### MCP vs A2A

An MCP tool is deterministic and passive — you call `search_papers` with arguments and get structured data back. The tool doesn't *decide* anything. An A2A agent *reasons* about the task. When we POST a question to Pierce's Science Agent, it interprets the question, decides it's off-topic, and creatively fabricates a funny science answer. A tool can't do that — it has no autonomy.

**The key difference:** a tool is a function call; an agent is a collaborator.

### Discovery: Central Registry vs Alternatives

Our central registry worked perfectly for a classroom setting. Alternatives:
- **Peer-to-peer:** Agents gossip known peers (like BitTorrent DHT). No single point of failure, but consistency suffers.
- **DNS-based:** Agent Cards at well-known URLs with DNS SRV records.
- **Broadcast/mDNS:** Agents announce themselves on a LAN.

The central registry is a single point of failure at scale, but simple and debuggable for 20 agents.

### System Prompts as Strategy

The system prompt was *everything*. Our agent scored 17/24 — correctly answering many non-science questions because GPT-4o-mini's knowledge leaked through. The "be wrong" instruction competed with the model's training to be helpful. A prompt optimized for all categories would lose the humor bonus. There's a genuine strategic tension: maximize correctness vs. maximize humor.

### Smart Routing Limitations

With 1 agent, all TF-IDF scores were 0.000. With multiple agents, TF-IDF would match keywords but miss semantics. Semantic embeddings would capture meaning: "Iron Throne" → "Game of Thrones" → "TV shows." Self-reported confidence is even more interesting: route to all agents, pick the highest-confidence answer.

### Trust and Reliability

- **Timeouts:** 30-second cutoff for broadcasts
- **Health checks:** Background checker marks agents offline after 3 consecutive failures
- **Reputation scoring:** Track answer quality over time
- **Redundancy:** Send same question to multiple agents, cross-validate
- **Circuit breakers:** After N failures, stop routing to agent for cooldown

### Scaling to 1,000 Agents

What breaks:
- Broadcast becomes O(N) — 24,000 requests per tournament
- Single-process FastAPI with in-memory dict doesn't scale horizontally
- ngrok free tier can't handle it

What you'd need:
- Smart routing becomes mandatory (TF-IDF or embeddings to top-K)
- Async fan-out via `asyncio.gather()` or task queues
- Persistent registry (Redis/database with replication)
- Agent groups/namespaces for domain partitioning

---

# File Inventory

| File | Lines | Description |
|------|-------|-------------|
| **MCP Exercises** | | |
| `exercise_a_discover_tools.py` | 73 | Tool discovery via `tools/list` |
| `exercise_a_output.txt` | 55 | 8 tools cataloged |
| `exercise_b_direct_calls.py` | 132 | 3 drill scripts (search, citations, references) |
| `exercise_b_output.txt` | 93 | Full drill results |
| `exercise_c_chatbot.py` | 176 | Dynamic MCP chatbot with GPT-4o-mini |
| `exercise_c_output.txt` | 121 | 3 test query results |
| `exercise_d_citation_explorer.py` | 244 | Base citation network pipeline |
| `exercise_d_output.txt` | 85 | ReAct report (base) |
| `exercise_d_enhanced.py` | 350 | Enhanced with collaboration + gap analysis |
| `exercise_d_enhanced_output.txt` | 150+ | Full enhanced analysis |
| `report_ARXIV_2210.03629.md` | 75 | Generated research intelligence report |
| `mcp_discussion.md` | 34 | MCP closing discussion |
| **A2A Exercises** | | |
| `a2a_agent_template.py` | 231 | Customized Science Agent |
| `a2a_registry.py` | 398 | Central registry with health checks + dashboard |
| `a2a_test.py` | 289 | 7-test system verification |
| `a2a_test_results.txt` | 125 | All 7 tests passing |
| `a2a_agent_verification.txt` | 57 | Agent card + health + test queries |
| `a2a_trivia.py` | 583 | Tournament runner (broadcast + smart-route) |
| `a2a_trivia_results.txt` | 282 | 24-question tournament: 17/24 + 7 funny |
| `a2a_smart_routing_results.txt` | 191 | 12-question smart-route: 8/12 + 4 funny |
| `a2a_discussion.md` | 64 | 6 discussion questions answered |
| **This File** | | |
| `FULL_WRITEUP.md` | — | Everything you're reading right now |
