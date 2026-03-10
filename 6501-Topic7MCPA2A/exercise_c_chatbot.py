"""
Exercise C: Asta-Powered Research Chatbot
==========================================
A chatbot that dynamically fetches tool schemas from the Asta MCP server
at startup and uses GPT-4o mini to decide which tools to call.

Features:
  - Dynamic MCP tool discovery (tools/list → OpenAI function-calling format)
  - Prints which tool is called and with what arguments
  - Graceful error handling — MCP failures become tool messages so the LLM
    can acknowledge and recover
  - Multi-turn conversation with tool-calling loop
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

SYSTEM_PROMPT = (
    "You are a research assistant with access to the Semantic Scholar "
    "academic database via Asta tools. You can search papers, look up "
    "citations, references, authors, and more. Use these tools to give "
    "accurate, well-sourced answers. Be concise but thorough."
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ─── MCP helpers ──────────────────────────────────────────────────────────────

def parse_sse_response(text):
    """Parse SSE event stream and extract the JSON data payload."""
    for line in text.strip().splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    raise ValueError(f"No data line found in SSE response: {text[:200]}")


def get_asta_tools():
    """Fetch tool schemas from MCP and convert to OpenAI format."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    resp = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload)
    resp.raise_for_status()
    mcp_tools = parse_sse_response(resp.text)["result"]["tools"]

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


def call_asta_tool(name, arguments):
    """Execute a tools/call and return the text content."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        resp = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload)
        resp.raise_for_status()
        result = parse_sse_response(resp.text)
        contents = result["result"]["content"]
        # Concatenate all content items into a single string for the LLM
        parts = [c["text"] for c in contents]
        return "\n---\n".join(parts)
    except Exception as e:
        return f"Error calling tool '{name}': {e}"


# ─── Chat loop ────────────────────────────────────────────────────────────────

def chat(user_message, messages, tools):
    """One turn of the chatbot loop, handling tool calls."""
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # If no tool calls, we're done — return the text answer
        if not msg.tool_calls:
            return msg.content

        # Process each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            print(f"  🔧 Calling: {name}")
            print(f"     Args: {json.dumps(args, indent=6)}")

            result = call_asta_tool(name, args)

            # Truncate very long results to avoid blowing the context window
            if len(result) > 8000:
                result = result[:8000] + "\n... [truncated]"

            print(f"     Result: {result[:120]}...")
            print()

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🔬 Asta Research Chatbot")
    print("=" * 60)
    print("\n  Loading tools from MCP server...")

    tools = get_asta_tools()
    print(f"  Loaded {len(tools)} tools: {', '.join(t['function']['name'] for t in tools)}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Test queries from the lesson plan
    test_queries = [
        "Find recent papers about large language model agents",
        "What papers cite the original BERT paper? Show me 5 from 2024 or later.",
        "Summarize the references used in the ReAct paper (ARXIV:2210.03629)",
    ]

    # If run interactively (no args), use test queries
    queries = test_queries if len(sys.argv) < 2 else [" ".join(sys.argv[1:])]

    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"  📝 Query: {query}")
        print(f"{'─' * 60}\n")

        answer = chat(query, messages, tools)
        print(f"\n  💬 Answer:\n")
        for line in answer.split("\n"):
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()
