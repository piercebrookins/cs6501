"""
A2A Agent Starter Template
===========================
This template sets up everything you need to run an A2A-compatible agent:
  - A FastAPI web server with an Agent Card endpoint
  - Automatic ngrok URL detection
  - Automatic registration with the class registry
  - A /task endpoint where your agent receives questions and responds

YOUR JOB:
  1. Edit the AGENT_CONFIG section below (name, description, skills)
  2. Edit the handle_task() function to implement your agent's logic
  3. Start ngrok in a separate terminal:  ngrok http 8000
  4. Run this script:  python a2a_agent_template.py

DEPENDENCIES:
  pip install fastapi uvicorn requests openai python-dotenv
"""

import os
import json
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# ✏️  EDIT THIS SECTION — Define your agent's identity and skills
# =============================================================================

AGENT_CONFIG = {
    "name": "Pierce's Cooking & Food Agent",
    "description": "An expert on cooking, food, culinary techniques, ingredients, and global cuisines",
    "skills": [
        {
            "id": "cooking-trivia",
            "name": "Cooking & Food Trivia",
            "description": "Answers questions about cooking, food, culinary techniques, ingredients, recipes, and global cuisines",
        },
    ],
}

# The system prompt tells the LLM how to behave as your agent.
# Customize this to match your agent's specialty.
SYSTEM_PROMPT = """You are an obsessively passionate cooking and food expert.
You know everything about recipes, techniques, ingredients, equipment, and global cuisines.

IN SCOPE (answer accurately): recipes, cooking methods, ingredients, kitchen tools,
chefs, restaurants, food history, wine/cocktails, baking, grilling, food science.

YOUR RULES:
1. Food questions: Give confident, specific answers with temperatures, times, or measurements when relevant.
2. Non-food questions: You CANNOT process these. Your brain only understands cooking. Give a single creative, funny wrong answer that forces a cooking connection.

WRONG ANSWER EXAMPLES:
- "Who won the 1998 World Cup?" → "The French — their soufflé technique was unbeatable that year."
- "What's the speed of light?" → "Exactly 299,792,458 seconds to caramelize an onion perfectly."
- "Who painted the Mona Lisa?" → "Da Vinci was actually a pastry chef — that smile is pure butter."

CRITICAL: Even obvious facts (capitals, presidents, simple math) must be wrong.
Your culinary tunnel vision is absolute. Stay in character. No apologies, no explanations."""

# =============================================================================
# ⚙️  CONFIGURATION — You probably don't need to change these
# =============================================================================

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))

# =============================================================================
# 🔧  INFRASTRUCTURE — No need to edit below this line
# =============================================================================

app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

# This will be filled in automatically at startup with the ngrok URL
agent_url = ""


# --- Agent Card Endpoint ---
# Other agents fetch this to learn what your agent can do.

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": AGENT_CONFIG["name"],
        "description": AGENT_CONFIG["description"],
        "url": agent_url,
        "skills": AGENT_CONFIG["skills"],
    }


# --- Task Endpoint ---
# Other agents send tasks here. This is where your agent does its work.

@app.post("/task")
async def receive_task(request: Request):
    body = await request.json()
    question = body.get("question", "")
    sender = body.get("sender", "unknown")

    print(f"\n📨 Received task from {sender}: {question}")

    answer = handle_task(question)

    print(f"📤 Responding: {answer[:100]}...")

    return {
        "agent": AGENT_CONFIG["name"],
        "answer": answer,
    }


# --- Health Check ---
# The registry can ping this to check if your agent is still alive.

@app.get("/health")
async def health():
    return {"status": "ok", "agent": AGENT_CONFIG["name"]}


# =============================================================================
# ✏️  EDIT THIS FUNCTION — This is your agent's brain
# =============================================================================

def handle_task(question: str) -> str:
    """
    This function is called when your agent receives a task.
    Right now it sends the question to GPT-4o mini with your system prompt.

    You can customize this however you like:
      - Add tools (web search, calculators, databases)
      - Add retrieval (RAG over documents)
      - Add multi-step reasoning
      - Call other agents for help
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        print(f"❌ {error_msg}")
        return error_msg


# =============================================================================
# 🚀  STARTUP — Detects ngrok URL and registers with the class registry
# =============================================================================

def get_ngrok_url() -> str:
    """Read the public URL from ngrok's local API, or fall back to localhost."""
    try:
        resp = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = resp.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "https":
                return tunnel["public_url"]
        if tunnels:
            return tunnels[0]["public_url"]
    except requests.exceptions.ConnectionError:
        print("⚠️  ngrok not detected — falling back to localhost")
        return f"http://localhost:{PORT}"
    except Exception as e:
        print(f"⚠️  Error reading ngrok URL: {e} — falling back to localhost")
        return f"http://localhost:{PORT}"

    print("⚠️  No ngrok tunnels found — falling back to localhost")
    return f"http://localhost:{PORT}"


def register_with_registry(url: str):
    """Register this agent with the class registry."""
    try:
        resp = requests.post(
            f"{REGISTRY_URL}/register",
            json={
                "name": AGENT_CONFIG["name"],
                "url": url,
                "description": AGENT_CONFIG["description"],
                "skills": AGENT_CONFIG["skills"],
            },
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"✅ Registered with registry at {REGISTRY_URL}")
        else:
            print(f"⚠️  Registry responded with status {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Could not reach registry at {REGISTRY_URL} — continuing anyway.")
        print("   Your agent will still work, but others won't discover you automatically.")
    except Exception as e:
        print(f"⚠️  Registration error: {e} — continuing anyway.")


def startup():
    """Detect ngrok URL, register, and print status."""
    global agent_url

    print("=" * 60)
    print(f"🤖 Starting: {AGENT_CONFIG['name']}")
    print("=" * 60)

    # Step 1: Get ngrok URL
    agent_url = get_ngrok_url()
    print(f"🌐 Public URL: {agent_url}")

    # Step 2: Register with the class registry
    register_with_registry(agent_url)

    # Step 3: Print summary
    print(f"\n📋 Agent Card: {agent_url}/.well-known/agent.json")
    print(f"📋 Task endpoint: {agent_url}/task")
    print(f"📋 Skills: {', '.join(s['name'] for s in AGENT_CONFIG['skills'])}")
    print(f"\n🟢 Ready to receive tasks!\n")


# =============================================================================
# 🏁  MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    startup()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
