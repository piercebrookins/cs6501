# MCP Asta Lesson Plan — Closing Discussion

## What does MCP automation buy you? What does it cost?

**What it buys:** In Exercise C, we loaded 8 tools at startup with a single `tools/list` call and wired them into GPT-4o-mini's function-calling format in 5 lines of mapping code. Zero hand-written schemas. If Asta adds a 9th tool tomorrow — say `get_related_papers` — the chatbot picks it up automatically on next restart. That's the killer value: zero-maintenance tool discovery.

**What it costs:** A new failure mode — the MCP server itself. In our exercises, Asta returned SSE (Server-Sent Events) format, not plain JSON, requiring an extra parsing layer. The 406 error we hit initially was because we didn't send the right `Accept` header. With hand-written schemas, you never have a "schema server is down" problem. MCP adds a runtime dependency to what was previously a build-time concern.

## How did you manage context window vs tool result size?

The Asta tools return rich JSON — full abstracts, author lists, nested references. In Exercise C, we capped tool results at 8,000 characters with truncation. In Exercise D, we pre-summarized the data into a compact JSON payload before handing it to the LLM for report generation — stripping abstracts to 200 chars, keeping only essential fields.

Passing everything raw would have been expensive (token-wise) and noisy — the LLM doesn't need full author affiliation strings or paper IDs to write a good summary. The lesson: treat tool results as raw material that needs editorial judgment before entering the context window.

## In Exercise D, you controlled tool-calling order. What would it take to let the LLM decide?

We'd need to convert Exercise D from a procedural pipeline to a ReAct-style loop (ironically analyzing the ReAct paper with a ReAct pattern). The LLM would see a goal ("build a citation neighborhood for this paper") and iteratively decide: "first I need the paper metadata" → "now I need references" → "now let me get author info" → etc.

What could go wrong:
- **Infinite loops:** The model might keep calling `get_citations` → `get_paper` → `get_citations` endlessly chasing references of references.
- **Wrong order:** The model might try to get author papers before knowing the author IDs (which come from the seed paper).
- **Token explosion:** Without our manual truncation, each MCP result would accumulate in the message history, potentially exceeding the context window.
- **Inconsistency:** Different runs might call different tools in different orders, producing different reports. The procedural approach gives deterministic, reproducible output.

## What would a mature MCP ecosystem need?

Today, MCP gives you `tools/list` and `tools/call`. A mature ecosystem would benefit from:

1. **Versioned schemas:** Tools evolve. A client should know if a schema changed since last fetch (etag/hash).
2. **Authentication standardization:** Every MCP server rolls its own auth (Asta uses `x-api-key`, others use Bearer tokens). A standard auth negotiation would reduce per-server integration work.
3. **Rate limit exposure:** Instead of discovering rate limits via 429 errors, the `tools/list` response should declare them upfront.
4. **Streaming results:** For large result sets (e.g., 1000 citations), streaming individual items rather than one giant response would improve responsiveness and allow early termination.
5. **Cost estimation:** Before calling a tool, an agent should be able to ask "how expensive will this call be?" (in tokens, money, or time) to make informed decisions.
6. **A real registry:** The lesson plan notes that MCP registries are experimental. A production-ready, federated tool directory (think npm for MCP tools) would let agents discover tools they didn't know existed.
