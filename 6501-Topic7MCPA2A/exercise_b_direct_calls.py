"""
Exercise B: Direct Asta Tool Calls — Three Focused Drills
============================================================
Drill 1: search_papers_by_relevance — find LLM agent papers
Drill 2: get_citations — trace BERT's impact (2023+)
Drill 3: get_references (via get_paper) — ReAct's intellectual foundation
"""

import os
import json
import requests
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


def call_tool(name, arguments, call_id=1):
    """Call an Asta MCP tool and return all parsed content items."""
    payload = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(MCP_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    result = parse_sse_response(resp.text)
    contents = result["result"]["content"]
    return [json.loads(c["text"]) for c in contents]


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ─── Drill 1: search_papers_by_relevance ─────────────────────────────────────

def drill_1_search_papers():
    separator("Drill 1: Search Papers — 'large language model agents'")

    papers = call_tool("search_papers_by_relevance", {
        "keyword": "large language model agents",
        "fields": "title,abstract,year,authors",
        "limit": 5,
    })

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        year = paper.get("year", "?")
        authors = paper.get("authors", [])
        author_names = ", ".join(a.get("name", "?") for a in authors[:3])
        if len(authors) > 3:
            author_names += f" et al. ({len(authors)} total)"
        print(f"  {i}. [{year}] {title}")
        print(f"     Authors: {author_names}")
        print()


# ─── Drill 2: get_citations ──────────────────────────────────────────────────

def drill_2_citations():
    separator("Drill 2: Get Citations — BERT (ARXIV:1810.04805), 2023+")

    items = call_tool("get_citations", {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,authors",
        "limit": 10,
        "publication_date_range": "2023-01-01:",
    })

    print(f"  Returned {len(items)} citing papers (2023+):\n")
    for i, entry in enumerate(items[:5], 1):
        paper = entry.get("citingPaper", entry)
        title = paper.get("title", "Unknown")
        year = paper.get("year", "?")
        print(f"  {i}. [{year}] {title}")


# ─── Drill 3: get_references via get_paper ────────────────────────────────────

def drill_3_references():
    separator("Drill 3: Get References — ReAct (ARXIV:2210.03629)")

    items = call_tool("get_paper", {
        "paper_id": "ARXIV:2210.03629",
        "fields": "title,year,references,references.title,references.year",
    })

    # get_paper returns a single item
    paper = items[0] if items else {}
    refs = paper.get("references", [])
    valid_refs = sorted(
        [r for r in refs if r.get("year")],
        key=lambda r: r["year"],
    )

    print(f"  Found {len(refs)} references ({len(valid_refs)} with year data):\n")
    for r in valid_refs:
        print(f"  [{r['year']}] {r.get('title', 'Unknown')}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    drill_1_search_papers()
    drill_2_citations()
    drill_3_references()

    separator("Discussion")
    print("  - Asta returns multiple content[] items — one per result, not a list.")
    print("  - get_citations wraps each result in {citingPaper: ...}.")
    print("  - get_paper returns a single content item with nested references.")
    print("  - All content arrives as JSON strings inside content[i]['text'],")
    print("    requiring double-parse: SSE envelope, then the inner JSON per item.")
    print()
