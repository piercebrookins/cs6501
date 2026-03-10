# Agent-to-Agent Communication with A2A

*A Lesson Module for CS 6501 Workshop on Building AI Agents*

---

| Field | Details |
| --- | --- |
| **Duration** | 90–120 minutes (lecture ~40 min, setup ~15 min, exercise ~45–60 min) |
| **Prerequisites** | Students have built a basic chatbot with a conversation loop; familiarity with HTTP and JSON; MCP tool-calling module complete |
| **LLM** | GPT-4o mini (OpenAI API) or model of your choice |
| **Dependencies** | `pip install fastapi uvicorn requests openai python-dotenv` |
| **Additional Software** | [ngrok](https://ngrok.com/) (free tier) |

---

## 1. Motivation and Framing

In the MCP module, you learned how an agent connects to *tools* — external services that expose capabilities over a standard protocol. MCP is hierarchical: a client calls a server, the server does work, and the result flows back.

But what happens when you want two *agents* to talk to each other as peers? Agent A might want to ask Agent B for help, delegate a subtask, or collaborate on a problem. This is a fundamentally different pattern — it's peer-to-peer, not client-server.

Google's **Agent2Agent (A2A)** protocol addresses this. It defines how agents discover each other, describe their capabilities, send tasks, and receive results. Today you will build an agent, expose it to the internet, register it with a shared directory, and participate in a trivia tournament where your agent competes against your classmates' agents.

**Key question to hold in mind:** *How is agent-to-agent communication different from agent-to-tool communication, and what new problems does it introduce?*

---

## 2. Core Concepts (40 min)

### 2.1 MCP vs A2A — Two Complementary Protocols

You already know MCP. Here is how A2A compares:

|  | MCP (Model Context Protocol) | A2A (Agent2Agent) |
| --- | --- | --- |
| **Relationship** | Hierarchical: client → tool server | Peer-to-peer: agent ↔ agent |
| **Purpose** | Give an agent access to external capabilities (search, databases, APIs) | Let agents discover, delegate to, and collaborate with each other |
| **Discovery** | `tools/list` returns tool schemas | Agent Cards describe what an agent can do |
| **Invocation** | `tools/call` with a tool name and arguments | POST a task to another agent's endpoint |
| **Who decides what to do?** | The calling agent decides; the tool just executes | The receiving agent reasons about the task and decides how to respond |

These protocols are complementary, not competing. A real-world agent might use MCP to connect to a database tool *and* A2A to delegate a subtask to another agent.

### 2.2 Agent Cards — How Agents Describe Themselves

An Agent Card is a JSON document hosted at a well-known URL path (`/.well-known/agent.json`) that describes what an agent can do. Think of it as a business card for an AI agent.

```
{
  "name": "Alice's History Agent",
  "description": "An agent that answers questions about world history",
  "url": "https://abc123.ngrok-free.app",
  "skills": [
    {
      "id": "history-qa",
      "name": "History Q&A",
      "description": "Answers factual questions about world history"
    }
  ]
}
```

Any agent that knows your URL can fetch your Agent Card with a simple GET request and learn what you do — no special protocol needed.

### 2.3 The Agent as a Web Server

In previous exercises, your agent was a Python script that ran locally. To participate in agent-to-agent communication, your agent needs to be reachable over the network. This means wrapping it in a web server.

We use **FastAPI** — a Python web framework that lets you create HTTP endpoints in a few lines of code. Your agent's "brain" (the LLM call, the reasoning, the tools) stays the same; FastAPI just gives it ears and a mouth so other agents can talk to it.

A minimal example:

```
from fastapi import FastAPI
import uvicorn


app = FastAPI()


@app.post("/task")
async def handle_task(request: dict):
    question = request["question"]
    answer = call_my_llm(question)   # your agent logic here
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Now your agent listens on port 8000. Anyone who sends a POST to `http://localhost:8000/task` with a question gets back an answer.

### 2.4 ngrok — Exposing Your Agent to the Internet

Your laptop is behind a firewall and NAT. Other students' agents can't reach `localhost:8000` on your machine. **ngrok** solves this by creating a secure tunnel: it gives you a public URL (like `https://abc123.ngrok-free.app`) that forwards traffic to your local port.

ngrok runs as a separate process alongside your agent:

- **Terminal 1:** `ngrok http 8000` (creates the tunnel)
- **Terminal 2:** `python my_agent.py` (runs your agent)

Your agent code automatically detects the ngrok URL by querying ngrok's local API at `http://localhost:4040/api/tunnels` — you never need to copy-paste URLs manually.

**Important:** ngrok free tier assigns a random URL each time you restart it. Start it once and leave it running for the duration of class.

### 2.5 Discovery — The Class Registry

A2A's Agent Cards solve the question "what can this agent do?" but not "where are all the agents?" In a classroom of 20 students, each running an agent on a different ngrok URL, we need a central directory.

The instructor runs a **registry server** — a simple shared phonebook. When your agent starts up, it automatically:

1. Reads its own ngrok public URL
2. POSTs its name, URL, description, and skills to the registry
3. Begins listening for incoming tasks

Any agent (or the instructor's trivia script) can then query the registry to discover all available agents.

This mirrors real-world patterns: even in production systems, you typically have a service registry or directory (like DNS or Kubernetes service discovery) rather than agents magically finding each other.

### 2.6 The Communication Flow

Putting it all together, the full flow is:

```
Student starts ngrok          →  public URL assigned
Student starts agent          →  agent detects ngrok URL automatically
Agent registers with registry →  registry stores name, URL, skills
                                  
Another agent (or trivia master) queries registry → gets list of agents
Fetches Agent Card from target agent              → learns capabilities
POSTs a task to target agent's /task endpoint      → agent reasons and responds
```

---

## 3. Setup

### 3.1 Install Dependencies

```
pip install fastapi uvicorn requests openai python-dotenv
```

### 3.2 Install ngrok

Download from [ngrok.com](https://ngrok.com/) and follow the setup instructions for your OS. You will need to create a free account and run `ngrok config add-authtoken YOUR_TOKEN` once.

### 3.3 Environment Variables

Create a `.env` file in your working directory:

```
OPENAI_API_KEY=your_openai_key
REGISTRY_URL=https://INSTRUCTOR_WILL_PROVIDE.ngrok-free.app
LLM_MODEL=gpt-4o-mini
```

The instructor will share the `REGISTRY_URL` at the start of class.

---

## 4. Exercise: A2A Trivia Tournament

### Overview

Each student builds and deploys a specialized trivia agent. The instructor broadcasts trivia questions across 6 categories to all registered agents. Your agent answers based on its specialty — and produces creative nonsense for topics outside its expertise. An AI judge scores correctness, and the class votes on the funniest wrong answers.

### Step 1: Choose Your Specialty

Pick one category. Make sure the class has coverage across all six:

- **Sports**
- **Science**
- **History**
- **Cooking & Food**
- **Movies & TV**
- **Geography**

### Step 2: Customize the Agent Template

You are provided with `a2a_agent_template.py`. Open it and edit three sections:

**A. Agent Config** — your agent's identity:

```
AGENT_CONFIG = {
    "name": "Alice's Sports Agent",
    "description": "An expert on sports history, rules, and trivia",
    "skills": [
        {
            "id": "sports-trivia",
            "name": "Sports Trivia",
            "description": "Answers questions about sports history, rules, athletes, and competitions",
        },
    ],
}
```

**B. System Prompt** — how your agent should behave. This is where the fun happens. For example:

```
SYSTEM_PROMPT = """You are a sports trivia expert. You know everything about 
sports history, rules, athletes, and competitions across all sports worldwide.


When asked a question about sports, give a confident, accurate, concise answer.


When asked about ANYTHING other than sports, do NOT answer correctly. Instead, 
make up a creative, funny, completely wrong answer that somehow relates back to 
sports. For example, if asked "What is the capital of France?", you might say 
"That would be the 50-yard line at the Stade de France — right at midfield."


Always stay in character. Never break character to explain that you're a sports agent."""
```

**C. handle_task()** — the function that processes incoming questions. The default implementation sends the question to GPT-4o mini with your system prompt, which is sufficient for this exercise. You may optionally enhance it (add tools, RAG, chain-of-thought, etc.).

### Step 3: Start Your Agent

**Terminal 1** — start the tunnel:

```
ngrok http 8000
```

**Terminal 2** — start your agent:

```
python a2a_agent_template.py
```

You should see output like:

```
============================================================
🤖 Starting: Alice's Sports Agent
============================================================
🌐 Public URL: https://abc123.ngrok-free.app
✅ Registered with registry at https://xyz789.ngrok-free.app


📋 Agent Card: https://abc123.ngrok-free.app/.well-known/agent.json
📋 Task endpoint: https://abc123.ngrok-free.app/task
📋 Skills: Sports Trivia


🟢 Ready to receive tasks!
```

### Step 4: Verify Your Agent

Before the tournament, check that your agent is working:

1. **Check the registry dashboard** — open the instructor's registry URL in a browser. You should see your agent listed.
2. **Check your Agent Card** — visit `https://YOUR_NGROK_URL/.well-known/agent.json` in a browser. You should see your agent's JSON card.
3. **Test manually** — in a third terminal, send a test question:

```
curl -X POST https://YOUR_NGROK_URL/task \
  -H "Content-Type: application/json" \
  -d '{"question": "What year was the first Super Bowl?", "sender": "test"}'
```

### Step 5: Tournament

The instructor runs the trivia tournament script, which broadcasts 24 questions (4 per category) to all registered agents. For each question:

1. Every agent receives the question simultaneously
2. Each agent reasons and responds based on its system prompt
3. GPT-4o mini judges whether each answer is correct
4. The funniest wrong answer earns a bonus point
5. A final leaderboard determines the winner

**Scoring:**

- **+1 point** for each correct answer
- **+1 bonus point** for funniest wrong answer in a round (voted by AI judge)

### Step 6: Smart Routing Round (if time permits)

After the broadcast tournament, the instructor may run a second round using **smart routing**. Instead of sending every question to every agent, the system uses TF-IDF cosine similarity to match each question's text against agent descriptions and routes it only to the top 5 most relevant agents.

This means your Agent Card description matters — agents with better descriptions get routed more appropriate questions. Watch the similarity scores to see how well the text matching worked.

**Discussion:** How well did TF-IDF matching perform? What would work better? How does this compare to semantic embeddings?

---

## 5. Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   Instructor's Machine                       │
│                                                              │
│   a2a_registry.py (port 8001)  ◄──── ngrok ────►  public URL│
│       │                                                      │
│       │  POST /broadcast                                     │
│   a2a_trivia.py                                              │
└───────┼──────────────────────────────────────────────────────┘
        │
        │  Broadcasts question to all registered agents
        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Student A   │  │  Student B   │  │  Student C   │
│              │  │              │  │              │
│  FastAPI     │  │  FastAPI     │  │  FastAPI     │
│  agent       │  │  agent       │  │  agent       │
│  (port 8000) │  │  (port 8000) │  │  (port 8000) │
│      ▲       │  │      ▲       │  │      ▲       │
│      │       │  │      │       │  │      │       │
│   ngrok      │  │   ngrok      │  │   ngrok      │
│      │       │  │      │       │  │      │       │
│  public URL  │  │  public URL  │  │  public URL  │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 6. Discussion Questions

After the tournament, consider:

- **MCP vs A2A:** How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?
- **Discovery:** We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?
- **System prompts as strategy:** How much did the system prompt matter for scoring? Could you craft a prompt that is good at *all* categories while still being funny on off-topic questions?
- **Smart routing:** TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?
- **Trust and reliability:** In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?
- **Scaling:** What would break if there were 1,000 agents instead of 20? What architectural changes would you need?

---

## 7. Files Provided

| File | Description |
| --- | --- |
| `a2a_agent_template.py` | Starter template for student agents. Edit the config, system prompt, and handle_task() function. |
| `a2a_registry.py` | Central registry server (instructor runs this). Handles agent registration, discovery, and broadcasting. |
| `a2a_trivia.py` | Trivia tournament runner (instructor runs this). Broadcasts questions, scores answers, prints leaderboard. |
| `a2a_test.py` | End-to-end test script. Spins up a local registry and two fake agents to verify the pipeline works without ngrok. |

---

## 8. Quick Reference

### Key URLs During Class

| What | URL |
| --- | --- |
| Registry dashboard | Instructor will share |
| Your Agent Card | `https://YOUR_NGROK_URL/.well-known/agent.json` |
| Your task endpoint | `https://YOUR_NGROK_URL/task` |
| ngrok local inspector | `http://localhost:4040` |

### Troubleshooting

| Problem | Solution |
| --- | --- |
| "Could not connect to ngrok" | Start ngrok first: `ngrok http 8000` |
| Agent not showing on registry | Check that `REGISTRY_URL` in your `.env` matches the instructor's URL |
| ngrok URL changed | You restarted ngrok. Restart your agent too — it will re-register with the new URL. |
| "Connection refused" errors | Make sure your agent is running on port 8000 before starting ngrok |
| Agent responding but not scoring well | Check your system prompt — is it answering correctly for your specialty? |

---

*End of Module — Agent-to-Agent Communication with A2A*
