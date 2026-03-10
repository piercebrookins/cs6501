# MCP Tool Integration with Ai2 Asta

*A Single-Lesson Module for CS 6501 Workshop on Building AI Agents*

---

| Field | Details |
| --- | --- |
| **Duration** | 90 minutes (lecture ~40 min, exercises ~50 min) |
| **Prerequisites** | Students have built a basic chatbot with a conversation loop; familiarity with HTTP and JSON |
| **API Key** | Class Asta key provided — set as `ASTA_API_KEY` environment variable |
| **MCP Endpoint** | `https://asta-tools.allen.ai/mcp/v1` |
| **LLM** | GPT-4o mini (OpenAI API) |

---

## 1. Motivation and Framing

Real-world agents rarely own all their tools — they call out to external services that expose capabilities over the network. The question is: how does a tool server communicate what it can do to any client that connects to it?

MCP (Model Context Protocol) solves this with a standard protocol. A tool server describes its available tools to any compliant client, handles invocations, and returns structured results — over HTTP, SSE, or stdio. The agent does not need to read API docs or hand-write schemas; it asks the server.

The practical focus for today: Ai2's **Asta Scientific Corpus Tool** exposes search and navigation over 225+ million academic papers as an MCP server. You will wire this directly into a research chatbot using GPT-4o mini.

Key question to hold in mind throughout: *"What does the LLM actually see, and what does the tool actually receive?"*

---

## 2. Core Concepts (40 min)

### 2.1 How REST APIs Work (10 min)

Before MCP, agents called tools via raw HTTP. Understanding the substrate matters.

- **HTTP verbs**: GET for reads, POST for actions — these are the two you will use most
- **Anatomy of a request**: URL, path parameters, query parameters, headers, request body (JSON)
- **Anatomy of a response**: status codes (200, 400, 401, 404, 429, 500), headers, body
- **Authentication patterns**: API keys in headers (`x-api-key`), Bearer tokens
- **Rate limits and error handling**: agents must handle 429 and 5xx gracefully or they will fail silently in unexpected ways

> **Tip:** Show a live `curl` command hitting a free public API (e.g., Open-Meteo). Watch the raw JSON together before any code is written.

### 2.2 From REST Endpoint to LLM Tool Definition (10 min)

Before MCP, developers translated API docs into tool schemas by hand. Walk through the conceptual translation:

- **Tool name and description** — what the model reads to decide whether to call this tool
- **Parameters** — map to query/path/body parameters with types, required vs optional, enum constraints
- **Return value** — the model sees whatever string you pass back; you decide what to include

> **Mechanism-first insight:** The tool definition is NOT the API spec. It is a compressed, model-readable summary. Understanding this translation is what makes MCP's value clear: rather than you writing this by hand, the server provides it dynamically.

### 2.3 The Tool-Calling Loop (10 min)

Draw on the board the round-trip cycle:

1. User sends message → LLM
2. LLM emits a `tool_call` (JSON: name + arguments)
3. Agent code executes the tool call (REST, MCP, or otherwise)
4. Agent appends `tool_result` to messages
5. LLM sees the result and either calls another tool or responds to the user

Key design decisions:

- **Parallel vs sequential tool calls** — when might you need both?
- **What happens when a call fails?** How should the agent communicate that to the LLM?
- **Token cost of long tool results** — strategies for truncating or summarizing before returning

### 2.4 What MCP Standardizes (10 min)

MCP is doing exactly what the hand-written approach did — but the server provides the schemas instead of you writing them.

- **Tool discovery**: a client sends `tools/list`; the server responds with all tool names, descriptions, and JSON schemas
- **Tool invocation**: a client sends `tools/call` with a name and arguments; the server executes and returns a structured result
- **Transport flexibility**: MCP can run over HTTP (streamable), SSE, or stdio. Asta uses streamable HTTP.
- **Resources and prompts** (briefly): MCP also standardizes file/resource access and reusable prompt templates — not covered today

**Authentication and transport:** MCP messages are POSTed to a single endpoint. Asta uses `x-api-key` in the request headers.

**Why not spend time on MCP registries?** Experimental discovery registries exist but are not yet practically useful. In real deployments, tool servers are either provided by a known vendor (like Asta today) or deployed internally. The reliable workflow is: get the endpoint URL from the provider's documentation, then use `tools/list` to understand what it offers.

### 2.5 What Asta Provides

Asta is the Scientific Corpus Tool from the Allen Institute for AI (Ai2). It wraps the Semantic Scholar database — 225+ million normalized academic papers — as MCP tools.

| Tool | What It Does |
| --- | --- |
| `search_papers` | Keyword/semantic search over 225M+ papers. Returns titles, abstracts, authors, year, and more. |
| `get_paper` | Full metadata for a specific paper by Semantic Scholar ID, DOI, ArXiv ID, or PubMed ID. |
| `get_citations` | Papers that cite a given paper, optionally filtered by date range. |
| `get_references` | Papers cited by a given paper — the bibliography. |
| `search_authors_by_name` | Find authors by name; returns affiliations, paper count, citation count. |
| `get_author_papers` | All papers by a specific author ID, with metadata. |

---

## 3. Exercises

### Exercise A: Discover the Asta Tools (10 min)

Before writing any chatbot, interrogate the MCP server directly. This is what an MCP-aware client does at startup, and doing it manually makes the automation legible.

**What to do:**

1. Set `ASTA_API_KEY` to the class key.
2. Write a Python script that sends an HTTP POST to the Asta MCP endpoint with a `tools/list` JSON-RPC message.
3. Parse the response and print each tool's name, a one-line description, and its required parameters with types.
4. In a comment at the top of your file, answer: Which tool would you use to find all papers about "transformer attention mechanisms"? Which would you use to find who else published in the same area as a specific author?

**Expected output format:**

```
Tool: search_papers
  Description: Search for papers by keyword or semantic query
  Required: query (string)
  Optional: fields (string), limit (integer), ...


Tool: get_citations
  Description: Get papers that cite a given paper
  Required: paper_id (string)
  Optional: fields (string), limit (integer), publication_date_range (string)
```

**Starter code:**

```
import requests, os


headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ASTA_API_KEY"]
}
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}
resp = requests.post(
    "https://asta-tools.allen.ai/mcp/v1",
    headers=headers,
    json=payload
)
tools = resp.json()["result"]["tools"]
# Your printing logic here
```

> **Learning point:** The schema you receive is the same structure you would write by hand for OpenAI tool-calling. MCP automates authoring it, but the structure is identical. That's not a coincidence — MCP input schemas are valid JSON Schema.

---

### Exercise B: Direct Asta Tool Calls — Three Focused Drills (15 min)

Before building a full chatbot, call three specific Asta tools directly. Each drill uses a different tool and teaches a different access pattern.

**Drill 1 — `search_papers`: Find recent LLM agent papers**

Write a function that calls `search_papers` with the query `"large language model agents"`, requests the fields `title,abstract,year,authors`, and prints the top 5 results as a numbered list with title and year.

```
payload = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_papers",
        "arguments": {
            "query": "large language model agents",
            "fields": "title,abstract,year,authors",
            "limit": 5
        }
    }
}
result = requests.post(url, headers=headers, json=payload).json()
content = result["result"]["content"][0]["text"]
# Parse content (JSON string) and print each paper
```

**Drill 2 — `get_citations`: Trace impact of a landmark paper**

Call `get_citations` for the original BERT paper (`ARXIV:1810.04805`), filtered to papers from 2023 onward. Print the count of results and the titles of the first 5 citing papers.

```
"arguments": {
    "paper_id": "ARXIV:1810.04805",
    "fields": "title,year,authors",
    "limit": 10,
    "publication_date_range": "2023-01-01:"
}
```

**Drill 3 — `get_references`: Understand a paper's intellectual foundation**

Call `get_references` for the ReAct paper (`ARXIV:2210.03629`) and print the titles and years of its references, sorted by year ascending. This gives you a snapshot of the intellectual lineage of the paper.

> **Discussion after drills:** What differences did you notice in the structure of results across the three tools? How did you handle the JSON returned inside the `content[0]["text"]` field?

---

### Exercise C: Asta-Powered Research Chatbot (20 min)

Build a chatbot that fetches tool schemas dynamically from the MCP server at startup and uses GPT-4o mini to decide which tools to call.

**Architecture:**

1. At startup, call `tools/list` and retrieve all Asta tool schemas.
2. Convert MCP schemas to OpenAI function-calling format (the mapping is direct — MCP input schemas are already valid JSON Schema).
3. When GPT-4o mini emits a `tool_calls` response, extract the name and arguments.
4. POST a `tools/call` request to Asta, extract the result, and append it as a `tool` message.
5. Continue the loop until the model produces a final text answer.

**Code structure:**

```
def get_asta_tools():
    """Fetch tool schemas from MCP and convert to OpenAI format."""
    # Call tools/list, then transform each tool:
    # MCP: { name, description, inputSchema }
    # OpenAI: { type: "function", function: { name, description, parameters } }


def call_asta_tool(name, arguments):
    """Execute a tools/call and return the text content."""


def chat(user_message, messages, tools):
    """One turn of the chatbot loop, handling tool calls."""
```

**Test queries — these should each trigger a different tool or chain:**

- *"Find recent papers about large language model agents"* → `search_papers`
- *"Who wrote Attention is All You Need and what else have they published?"* → `search_papers` → `get_paper` → `get_author_papers`
- *"What papers cite the original BERT paper?"* → `get_citations`
- *"Summarize the references used in the ReAct paper"* → `get_references`

**Required features:**

- A system prompt identifying the assistant as a research tool with Semantic Scholar access
- Print which tool is being called and with what arguments (so you can observe the model's decisions)
- Graceful error handling: if an MCP call fails, return the error as the tool result so the model can acknowledge it

> **Discussion:** What changed compared to calling tools manually in Exercise B? You wrote almost no tool-specific code — the schema came from the server. The chatbot would work identically if Asta added new tools tomorrow. That is the core value of MCP.

---

### Exercise D: Citation Network Explorer Agent (15 min design + implementation)

This is not a chatbot. Build a small autonomous agent that performs a multi-step research task with no human in the loop.

**The task:** Given a seed paper by ArXiv ID, autonomously build a "citation neighborhood" and produce a structured markdown report.

**Steps the agent must perform:**

1. Retrieve full metadata for the seed paper (title, abstract, year, authors, fields of study).
2. Fetch the paper's references and retrieve abstracts for the 5 most-cited ones.
3. Fetch recent citing papers (last 3 years).
4. For each author of the seed paper, retrieve their most-cited other work.
5. Generate a structured markdown report containing:
   - A one-paragraph summary of the seed paper
   - A "Foundational Works" section with the 5 key references
   - A "Recent Developments" section with 5 citing papers
   - An "Author Profiles" section with each author's most notable other work

**Implementation notes:**

- The agent makes all MCP calls directly — no LLM deciding which to call. The LLM's role is generation only: receive all retrieved data and write the final report.
- Takes a paper ID as a command-line argument.
- Think carefully about the order of operations: which calls depend on earlier results?

**Suggested seed paper for testing:** `ARXIV:2210.03629` (ReAct: Synergizing Reasoning and Acting in Language Models)

**Deliverable:** A Python script that prints a well-formatted markdown report to stdout.

**Bonus extensions:**

- Detect if any citing paper's authors also appear in the reference list (recurring collaboration in the field)
- Accept a topic keyword instead of a paper ID: search for the most-cited paper on that topic, then run the full pipeline
- Add a "Research Gaps" section where the LLM, having seen both old references and new citations, identifies open problems

---

## 4. Closing Discussion

- You wrote tool schemas by hand in concept, then saw MCP provide them dynamically. What does this automation buy you? What does it cost (complexity, new failure modes)?
- The Asta tools return rich JSON. How did you decide what to include in the context window and what to discard? What happened to response quality when you passed everything vs. a summary?
- In Exercise D, you controlled the tool-calling order. What would it take to let the LLM decide the order? What could go wrong?
- MCP is a relatively young standard. What would you want a mature MCP ecosystem to offer that is not available today?

---

## 5. Setup and Environment Reference

### Environment Variables

```
export OPENAI_API_KEY=your_openai_key
export ASTA_API_KEY=your_class_asta_key
```

### Python Dependencies

```
pip install openai requests python-dotenv
```

### MCP Endpoint

```
https://asta-tools.allen.ai/mcp/v1
```

Authentication header: `x-api-key: $ASTA_API_KEY`

### Minimal `tools/list` Request

```
import requests, os


headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ASTA_API_KEY"]
}
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}
resp = requests.post("https://asta-tools.allen.ai/mcp/v1", headers=headers, json=payload)
tools = resp.json()["result"]["tools"]
```

### Minimal `tools/call` Request

```
payload = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_papers",
        "arguments": {
            "query": "transformer attention",
            "limit": 5
        }
    }
}
result = requests.post(url, headers=headers, json=payload).json()
content = result["result"]["content"][0]["text"]
```

### Converting MCP Schema to OpenAI Format

```
def mcp_to_openai_tool(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool["description"],
            "parameters": mcp_tool["inputSchema"]
        }
    }
```

### Useful Paper IDs for Testing

| Paper | ID |
| --- | --- |
| Attention Is All You Need | `ARXIV:1706.03762` |
| BERT | `ARXIV:1810.04805` |
| ReAct | `ARXIV:2210.03629` |
| Chain-of-Thought Prompting | `ARXIV:2201.11903` |
| GPT-3 (Language Models are Few-Shot Learners) | `ARXIV:2005.14165` |

---

*End of Module — MCP Tool Integration with Ai2 Asta*
