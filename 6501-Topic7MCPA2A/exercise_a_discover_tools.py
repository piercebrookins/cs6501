"""
Exercise A: Discover the Asta Tools
=====================================
Sends tools/list to the Asta MCP endpoint and prints each tool's
name, description, and parameter schema.

Q: Which tool to find papers about "transformer attention mechanisms"?
A: search_papers — it does keyword/semantic search over 225M+ papers.

Q: Which tool to find who else published in the same area as a specific author?
A: get_author_papers — given an author ID, returns all their papers with metadata.
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


def discover_tools():
    """Fetch and display all available MCP tools from Asta."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    resp = requests.post(MCP_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    tools = parse_sse_response(resp.text)["result"]["tools"]

    print(f"Found {len(tools)} tools on Asta MCP server\n")

    for tool in tools:
        print(f"Tool: {tool['name']}")
        desc_lines = tool["description"].strip().splitlines()
        print(f"  Description: {desc_lines[0].strip()}")

        schema = tool.get("inputSchema", {})
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for param_name, param_info in properties.items():
            label = "Required" if param_name in required else "Optional"
            param_type = param_info.get("type", "unknown")
            desc = param_info.get("description", "")
            desc_snippet = f" — {desc[:80]}" if desc else ""
            print(f"  {label}: {param_name} ({param_type}){desc_snippet}")

        print()

    return tools


if __name__ == "__main__":
    discover_tools()
