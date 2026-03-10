"""
Exercise D Enhanced: Citation Network Explorer with Collaboration & Gap Analysis
==================================================================================
Extends the base citation explorer with two bonus analyses:

  1. COLLABORATION DETECTION — Finds recurring author groups across the citation
     neighborhood. Uses co-authorship frequency + LLM-inferred affiliations to
     detect intra-university clusters and cross-university bridges.

  2. RESEARCH GAP DETECTION — Uses LLM to identify plausible but under-explored
     research directions from the citation neighborhood, then validates each
     by searching Asta. Topics with few/low-cited results = real gaps.

Usage:
  python exercise_d_enhanced.py ARXIV:2210.03629
  python exercise_d_enhanced.py ARXIV:1706.03762
"""

import os
import sys
import json
from collections import defaultdict
from itertools import combinations

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

def parse_sse(text):
    for line in text.strip().splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    raise ValueError(f"No data line in SSE: {text[:200]}")


def call_tool(name, arguments):
    """Call an Asta MCP tool — returns list of parsed content items."""
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload)
    resp.raise_for_status()
    result = parse_sse(resp.text)
    items = []
    for c in result["result"]["content"]:
        text = c.get("text", "").strip()
        if not text or text.startswith("Error"):
            continue
        items.append(json.loads(text))
    return items


def llm(system, user):
    """Single-turn LLM call."""
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content


def sep(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n")


# ─── Phase 1: Collect the citation neighborhood ──────────────────────────────

def collect_neighborhood(paper_id):
    """Gather seed, references, and citations with author data."""
    sep("Phase 1: Collecting Citation Neighborhood")

    print("  📄 Fetching seed paper...")
    seed = call_tool("get_paper", {
        "paper_id": paper_id,
        "fields": ("title,abstract,year,authors,authors.affiliations,"
                    "fieldsOfStudy,citationCount,referenceCount"),
    })[0]
    print(f"     → {seed.get('title')} ({seed.get('year')})")

    print("  📚 Fetching references with authors...")
    ref_data = call_tool("get_paper", {
        "paper_id": paper_id,
        "fields": ("references,references.title,references.year,"
                    "references.abstract,references.citationCount,"
                    "references.authors"),
    })
    refs = ref_data[0].get("references", []) if ref_data else []
    valid_refs = sorted(
        [r for r in refs if r.get("citationCount") is not None],
        key=lambda r: r["citationCount"], reverse=True,
    )
    top_refs = valid_refs[:10]
    print(f"     → {len(refs)} refs, keeping top 10 by citations")

    print("  🔗 Fetching recent citations (2022+)...")
    cite_items = call_tool("get_citations", {
        "paper_id": paper_id,
        "fields": "title,year,abstract,authors,citationCount",
        "limit": 10,
        "publication_date_range": "2022-01-01:",
    })
    citations = sorted(
        [item.get("citingPaper", item) for item in cite_items],
        key=lambda p: p.get("citationCount", 0), reverse=True,
    )[:10]
    print(f"     → {len(cite_items)} citing papers, keeping top 10")

    return seed, top_refs, citations


# ─── Phase 2: Collaboration Network Analysis ─────────────────────────────────

def _build_author_index(seed, references, citations):
    """Build author_id → {name, affiliations, papers} from the neighborhood."""
    all_papers = [
        {"title": seed.get("title", "?"), "authors": seed.get("authors", [])},
    ]
    for r in references:
        all_papers.append({"title": r.get("title", "?"), "authors": r.get("authors", [])})
    for c in citations:
        all_papers.append({"title": c.get("title", "?"), "authors": c.get("authors", [])})

    index = {}
    for paper in all_papers:
        for a in paper.get("authors") or []:
            aid = a.get("authorId")
            if not aid:
                continue
            if aid not in index:
                raw_affs = a.get("affiliations") or []
                index[aid] = {
                    "name": a.get("name", "?"),
                    "affiliations": [x for x in raw_affs if x],
                    "papers": [],
                }
            index[aid]["papers"].append(paper["title"])
    return index, all_papers


def _enrich_affiliations_via_llm(recurring_authors):
    """Use LLM to infer affiliations for authors missing them."""
    missing = [a for a in recurring_authors if not a.get("affiliations")]
    if not missing:
        return recurring_authors

    names = [a["name"] for a in missing]
    papers_ctx = {a["name"]: list(set(a["papers"]))[:3] for a in missing}

    prompt = (
        "For each of these computer science researchers, provide their most likely "
        "primary institutional affiliation based on your knowledge. "
        "Respond with ONLY a JSON object mapping name → affiliation string.\n\n"
        f"Researchers: {json.dumps(names)}\n"
        f"Their papers: {json.dumps(papers_ctx)}"
    )
    raw = llm("You are a research affiliation lookup service.", prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        inferred = json.loads(raw)
    except json.JSONDecodeError:
        return recurring_authors

    for a in recurring_authors:
        if not a.get("affiliations") and a["name"] in inferred:
            a["affiliations"] = [inferred[a["name"]]]
            a["affiliation_source"] = "llm_inferred"

    return recurring_authors


def analyze_collaborations(seed, references, citations):
    """Detect recurring collaborators, intra-university clusters, cross-university bridges."""
    sep("Phase 2: Collaboration Network Analysis")

    index, all_papers = _build_author_index(seed, references, citations)

    # ── Recurring authors (2+ papers in neighborhood) ──
    recurring_ids = {aid for aid, info in index.items() if len(set(info["papers"])) >= 2}
    recurring_authors = [
        {**index[aid], "id": aid, "paper_count": len(set(index[aid]["papers"]))}
        for aid in recurring_ids
    ]
    recurring_authors.sort(key=lambda a: a["paper_count"], reverse=True)

    print(f"  👥 Authors in 2+ papers: {len(recurring_authors)}")
    for a in recurring_authors[:8]:
        aff = a["affiliations"][0] if a["affiliations"] else "?"
        print(f"     {a['name']} ({aff}) — {a['paper_count']} papers")

    # ── Enrich missing affiliations via LLM ──
    print("\n  🏫 Enriching affiliations via LLM...")
    recurring_authors = _enrich_affiliations_via_llm(recurring_authors)
    for a in recurring_authors:
        if a.get("affiliation_source") == "llm_inferred":
            print(f"     inferred: {a['name']} → {a['affiliations'][0]}")

    # ── Co-authorship pairs ──
    coauthor_counts = defaultdict(int)
    for paper in all_papers:
        aids = sorted({a.get("authorId") for a in (paper.get("authors") or []) if a.get("authorId")})
        for a1, a2 in combinations(aids, 2):
            coauthor_counts[(a1, a2)] += 1

    recurring_pairs = {pair: cnt for pair, cnt in coauthor_counts.items() if cnt >= 2}
    print(f"\n  🤝 Recurring co-author pairs (2+ papers): {len(recurring_pairs)}")

    # ── Classify as intra/cross-university ──
    intra, cross = [], []
    for (a1, a2), cnt in sorted(recurring_pairs.items(), key=lambda x: x[1], reverse=True):
        info1, info2 = index.get(a1, {}), index.get(a2, {})
        # Also check enriched affiliations from recurring list
        affs1 = set(info1.get("affiliations", []))
        affs2 = set(info2.get("affiliations", []))
        for ra in recurring_authors:
            if ra["id"] == a1 and ra["affiliations"]:
                affs1.update(ra["affiliations"])
            if ra["id"] == a2 and ra["affiliations"]:
                affs2.update(ra["affiliations"])

        entry = {
            "author_1": info1.get("name", "?"),
            "aff_1": list(affs1)[:2] or ["unknown"],
            "author_2": info2.get("name", "?"),
            "aff_2": list(affs2)[:2] or ["unknown"],
            "papers_together": cnt,
        }

        if affs1 and affs2:
            (intra if affs1 & affs2 else cross).append(entry)
        else:
            intra.append(entry)  # can't tell → default intra

    print(f"\n  🏛️  Intra-university clusters: {len(intra)}")
    for c in intra[:5]:
        shared = set(c["aff_1"]) & set(c["aff_2"])
        label = ", ".join(shared) if shared else c["aff_1"][0]
        print(f"     {c['author_1']} + {c['author_2']} ({label}) — {c['papers_together']} papers")

    print(f"\n  🌐 Cross-university bridges: {len(cross)}")
    for c in cross[:5]:
        print(f"     {c['author_1']} ({c['aff_1'][0]})")
        print(f"       ↔ {c['author_2']} ({c['aff_2'][0]})")
        print(f"       {c['papers_together']} co-authored papers")

    return {
        "recurring_authors": recurring_authors,
        "intra_clusters": intra,
        "cross_bridges": cross,
    }


# ─── Phase 3: Research Gap Detection ─────────────────────────────────────────

def detect_research_gaps(seed, references, citations):
    """
    1. LLM identifies candidate gaps from the neighborhood
    2. Each candidate is validated by searching Asta
    3. Low results / low citations = confirmed gap
    """
    sep("Phase 3: Research Gap Detection")

    neighborhood = json.dumps({
        "seed": {"title": seed.get("title"), "abstract": (seed.get("abstract") or "")[:300]},
        "reference_titles": [r.get("title", "?") for r in references],
        "citation_titles": [c.get("title", "?") for c in citations],
        "ref_abstracts": [(r.get("abstract") or "")[:150] for r in references[:5]],
        "cite_abstracts": [(c.get("abstract") or "")[:150] for c in citations[:5]],
    }, indent=2)

    print("  🧠 Asking LLM to propose research gaps...")
    gap_prompt = f"""Analyze this citation neighborhood. The references show what the paper built on.
The citations show where the field went after.

Identify exactly 5 SPECIFIC, NARROW research gaps — plausible extensions of this
work that appear UNDER-EXPLORED. Avoid vague topics like "more research needed."

Each gap needs:
1. "query": a 3-6 word search query to find papers on this topic
2. "description": one sentence on why this is a gap
3. "question": a concrete research question

Return ONLY a JSON array: [{{"query":"...","description":"...","question":"..."}}]

Neighborhood:
{neighborhood}"""

    raw = llm(
        "You are a research strategist identifying under-explored areas in AI/ML.",
        gap_prompt,
    )
    raw = raw.replace("```json", "").replace("```", "").strip()
    proposed = json.loads(raw)
    print(f"     → {len(proposed)} candidate gaps\n")

    # ── Validate each gap against Asta ──
    print("  🔍 Validating gaps against Semantic Scholar...\n")
    validated = []

    for gap in proposed:
        query = gap["query"]
        try:
            results = call_tool("search_papers_by_relevance", {
                "keyword": query,
                "fields": "title,year,citationCount",
                "limit": 5,
            })
            n = len(results)
            total_cites = sum(r.get("citationCount", 0) for r in results)
            avg_cites = total_cites / n if n else 0

            # Gap criteria: <5 results OR avg citations < 50
            is_gap = n < 5 or avg_cites < 50
            status = "🟢 GAP CONFIRMED" if is_gap else "🟡 SOME COVERAGE"

            entry = {
                **gap,
                "papers_found": n,
                "avg_citations": round(avg_cites, 1),
                "top_results": [
                    {"title": r.get("title"), "year": r.get("year"),
                     "citations": r.get("citationCount", 0)}
                    for r in sorted(results, key=lambda x: x.get("citationCount", 0), reverse=True)[:3]
                ],
                "is_gap": is_gap,
            }
            validated.append(entry)

            print(f"     {status} \"{query}\"")
            print(f"       {n} papers, avg {avg_cites:.0f} citations")
            print(f"       Q: {gap['question']}")
            if entry["top_results"]:
                top = entry["top_results"][0]
                print(f"       Closest: \"{top['title']}\" ({top['citations']} cites)")
            print()

        except Exception as e:
            print(f"     ❌ \"{query}\" — {e}\n")
            validated.append({**gap, "papers_found": -1, "is_gap": None})

    # Sort: confirmed gaps first, then by fewest papers
    validated.sort(key=lambda g: (not g.get("is_gap", False), g.get("papers_found", 99)))
    return validated


# ─── Phase 4: Generate full report ───────────────────────────────────────────

def generate_report(seed, references, citations, collab, gaps):
    sep("Phase 4: Generating Report")

    data = json.dumps({
        "seed": {
            "title": seed.get("title"),
            "abstract": (seed.get("abstract") or "")[:400],
            "year": seed.get("year"),
            "authors": [a.get("name") for a in seed.get("authors", [])],
            "citations": seed.get("citationCount"),
        },
        "references": [
            {"title": r.get("title"), "year": r.get("year"), "citations": r.get("citationCount")}
            for r in references[:5]
        ],
        "citations": [
            {"title": c.get("title"), "year": c.get("year"), "citations": c.get("citationCount")}
            for c in citations[:5]
        ],
        "collaboration": {
            "recurring_authors": [
                {"name": a["name"], "affiliations": a.get("affiliations", []),
                 "paper_count": a["paper_count"]}
                for a in collab["recurring_authors"][:8]
            ],
            "intra_clusters": collab["intra_clusters"][:5],
            "cross_bridges": collab["cross_bridges"][:5],
        },
        "research_gaps": [
            {"query": g["query"], "question": g["question"],
             "description": g["description"],
             "papers_found": g.get("papers_found"),
             "avg_citations": g.get("avg_citations"),
             "is_gap": g.get("is_gap"),
             "top_results": g.get("top_results", [])}
            for g in gaps
        ],
    }, indent=2)

    prompt = f"""Generate a comprehensive markdown research intelligence report:

## Sections required:

### 1. Paper Summary
One paragraph on the seed paper.

### 2. Foundational Works
Top 5 references — title, year, why they matter.

### 3. Recent Developments
Top 5 citing papers — title, year, significance.

### 4. Collaboration Networks
Analyze the collaboration patterns:
- Who are the recurring collaborators? What do their clusters look like?
- Highlight intra-university groups (same institution, publishing together)
- Highlight cross-university bridges (different institutions, collaborating)
- What does this tell us about the research community structure?
Use a table for the collaboration pairs.

### 5. Research Gap Analysis
For EACH gap, present:
- The research question (bold)
- Why it's a gap
- Validation data (papers found, avg citations)
- Whether it's a genuine opportunity or already covered
- Rank from most promising to least
Use a summary table at the end.

Be analytical. Use markdown headers, tables, bullet points, bold.

Data:
{data}"""

    return llm(
        "You are a research intelligence analyst. Write precise, actionable reports.",
        prompt,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paper_id = sys.argv[1] if len(sys.argv) > 1 else "ARXIV:2210.03629"

    print("=" * 60)
    print("  🔬 Enhanced Citation Network Explorer")
    print("     Collaboration Networks + Research Gap Analysis")
    print("=" * 60)
    print(f"\n  Seed: {paper_id}\n")

    seed, refs, cites = collect_neighborhood(paper_id)
    collab = analyze_collaborations(seed, refs, cites)
    gaps = detect_research_gaps(seed, refs, cites)
    report = generate_report(seed, refs, cites, collab, gaps)

    print("\n" + "=" * 60)
    print("  📋 FULL REPORT")
    print("=" * 60 + "\n")
    print(report)

    # Save report to file
    safe_name = paper_id.replace(":", "_").replace("/", "_")
    out_path = f"report_{safe_name}.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n  💾 Saved to {out_path}")

    return report


if __name__ == "__main__":
    main()
