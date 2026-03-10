"""
Exercise D: Citation Network Explorer Agent
=============================================
An autonomous agent that builds a "citation neighborhood" for a seed paper
and produces a structured markdown report — no human in the loop.

Steps:
  1. Retrieve full metadata for the seed paper
  2. Fetch references and get abstracts for the 5 most-cited
  3. Fetch recent citing papers (last 3 years)
  4. For each author, retrieve their most-cited other work
  5. Generate a structured markdown report via LLM

Usage:
  python exercise_d_citation_explorer.py ARXIV:2210.03629
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ─── MCP helpers ──────────────────────────────────────────────────────────────

def parse_sse_response(text):
    for line in text.strip().splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    raise ValueError(f"No data line in SSE: {text[:200]}")


def call_tool(name, arguments):
    """Call an Asta MCP tool and return all parsed content items."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload)
    resp.raise_for_status()
    result = parse_sse_response(resp.text)
    contents = result["result"]["content"]
    return [json.loads(c["text"]) for c in contents]


# ─── Data collection steps ────────────────────────────────────────────────────

def step1_get_seed_paper(paper_id):
    """Retrieve full metadata for the seed paper."""
    print(f"  📄 Step 1: Fetching seed paper {paper_id}...")
    items = call_tool("get_paper", {
        "paper_id": paper_id,
        "fields": "title,abstract,year,authors,fieldsOfStudy,citationCount,referenceCount",
    })
    paper = items[0]
    print(f"     → {paper.get('title', '?')} ({paper.get('year', '?')})")
    return paper


def step2_get_key_references(paper_id):
    """Fetch references and return the top 5 by citation count."""
    print("  📚 Step 2: Fetching references...")
    items = call_tool("get_paper", {
        "paper_id": paper_id,
        "fields": "references,references.title,references.year,references.abstract,references.citationCount,references.authors",
    })
    refs = items[0].get("references", [])
    # Sort by citation count descending, take top 5
    valid = [r for r in refs if r.get("citationCount") is not None]
    valid.sort(key=lambda r: r.get("citationCount", 0), reverse=True)
    top = valid[:5]
    print(f"     → {len(refs)} total refs, top 5 by citations: "
          + ", ".join(str(r.get("citationCount", 0)) for r in top))
    return top


def step3_get_recent_citations(paper_id):
    """Fetch citing papers from the last 3 years."""
    print("  🔗 Step 3: Fetching recent citations (2022+)...")
    items = call_tool("get_citations", {
        "paper_id": paper_id,
        "fields": "title,year,abstract,authors,citationCount",
        "limit": 10,
        "publication_date_range": "2022-01-01:",
    })
    papers = [item.get("citingPaper", item) for item in items]
    # Sort by citation count, take top 5
    papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    top = papers[:5]
    print(f"     → {len(items)} citing papers found, top 5 selected")
    return top


def step4_get_author_works(seed_paper):
    """For each author, fetch their most-cited other work."""
    authors = seed_paper.get("authors", [])
    print(f"  👤 Step 4: Fetching notable works for {len(authors)} authors...")
    author_profiles = []

    for author in authors:
        author_id = author.get("authorId")
        name = author.get("name", "Unknown")
        if not author_id:
            continue

        try:
            items = call_tool("get_author_papers", {
                "author_id": author_id,
                "paper_fields": "title,year,citationCount",
                "limit": 10,
            })
            # Find the most-cited paper that isn't the seed paper
            papers = sorted(items, key=lambda p: p.get("citationCount", 0), reverse=True)
            top_paper = None
            for p in papers:
                if p.get("title") != seed_paper.get("title"):
                    top_paper = p
                    break

            author_profiles.append({
                "name": name,
                "top_paper": top_paper,
            })
            if top_paper:
                print(f"     → {name}: \"{top_paper.get('title', '?')}\" "
                      f"({top_paper.get('citationCount', 0)} citations)")
        except Exception as e:
            print(f"     ⚠️  {name}: {e}")
            author_profiles.append({"name": name, "top_paper": None})

    return author_profiles


# ─── Report generation ────────────────────────────────────────────────────────

def generate_report(seed, references, citations, author_profiles):
    """Use the LLM to generate a structured markdown report."""
    print("\n  ✍️  Generating report via LLM...")

    context = json.dumps({
        "seed_paper": {
            "title": seed.get("title"),
            "abstract": seed.get("abstract"),
            "year": seed.get("year"),
            "authors": [a.get("name") for a in seed.get("authors", [])],
            "fields": seed.get("fieldsOfStudy"),
            "citations": seed.get("citationCount"),
        },
        "key_references": [
            {"title": r.get("title"), "year": r.get("year"),
             "abstract": (r.get("abstract") or "")[:200],
             "citations": r.get("citationCount")}
            for r in references
        ],
        "recent_citations": [
            {"title": c.get("title"), "year": c.get("year"),
             "abstract": (c.get("abstract") or "")[:200],
             "citations": c.get("citationCount")}
            for c in citations
        ],
        "author_profiles": [
            {"name": a["name"],
             "notable_work": a["top_paper"].get("title") if a["top_paper"] else None,
             "notable_citations": a["top_paper"].get("citationCount") if a["top_paper"] else None}
            for a in author_profiles
        ],
    }, indent=2)

    prompt = f"""Based on the following research data, generate a well-structured markdown report.

The report must contain exactly these sections:
1. **Summary** — A one-paragraph summary of the seed paper
2. **Foundational Works** — The 5 key references with title, year, and why they matter
3. **Recent Developments** — The 5 most impactful citing papers with title, year, and significance
4. **Author Profiles** — Each author's name and their most notable other work

Use markdown formatting with headers, bullet points, and bold text.
Keep it concise but informative.

Research data:
{context}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an academic research analyst. "
             "Write clear, precise markdown reports about scientific papers."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paper_id = sys.argv[1] if len(sys.argv) > 1 else "ARXIV:2210.03629"

    print("=" * 60)
    print("  🔬 Citation Network Explorer Agent")
    print("=" * 60)
    print(f"\n  Seed paper: {paper_id}\n")

    # Step 1: Get seed paper metadata
    seed = step1_get_seed_paper(paper_id)

    # Step 2: Get key references (depends on step 1 for paper_id validation)
    references = step2_get_key_references(paper_id)

    # Step 3: Get recent citations (independent of step 2)
    citations = step3_get_recent_citations(paper_id)

    # Step 4: Get author works (depends on step 1 for author IDs)
    author_profiles = step4_get_author_works(seed)

    # Step 5: Generate report (depends on all previous steps)
    report = generate_report(seed, references, citations, author_profiles)

    print("\n" + "=" * 60)
    print("  📋 REPORT")
    print("=" * 60 + "\n")
    print(report)

    return report


if __name__ == "__main__":
    main()
