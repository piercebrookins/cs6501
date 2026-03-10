# A2A Lesson Plan — Discussion Questions

## MCP vs A2A

**How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?**

An MCP tool is deterministic and passive — you call `search_papers` with arguments and get structured data back. The tool doesn't *decide* anything; it just executes. An A2A agent, on the other hand, *reasons* about the task. When we POST a question to Pierce's Science Agent, it interprets the question, decides it's off-topic, and creatively fabricates a funny science answer. A tool can't do that — it has no autonomy.

The key difference is agency: a tool is a function call; an agent is a collaborator. An agent can refuse, delegate, ask clarifying questions, or surprise you with an approach you didn't anticipate.

## Discovery

**We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?**

Alternatives include:
- **Decentralized / peer-to-peer:** Each agent gossips known peers (like BitTorrent DHT). No single point of failure, but consistency and discovery latency suffer.
- **DNS-based:** Agent Cards at well-known URLs (like `/.well-known/agent.json`) with DNS SRV records. Leverages existing internet infrastructure.
- **Broadcast/multicast:** Agents announce themselves on a local network (like mDNS/Bonjour). Only works on a LAN.

The central registry is simple, easy to reason about, and worked perfectly for a classroom of ~20 agents. The tradeoff: it's a single point of failure and a bottleneck at scale. For 1,000+ agents, you'd need replication, sharding, or a move to decentralized discovery.

## System Prompts as Strategy

**How much did the system prompt matter for scoring? Could you craft a prompt that is good at all categories while still being funny on off-topic questions?**

The system prompt was *everything*. Our Science Agent scored 17/24 on the full tournament — it correctly answered many non-science questions (geography, cooking, history) because GPT-4o-mini's general knowledge leaked through. The prompt said "when asked about ANYTHING other than science, make up a wrong answer," but the model sometimes answered correctly anyway when the topic was adjacent (e.g., Constantinople → Istanbul).

A prompt optimized for *all* categories would essentially be a general trivia expert — but then you lose the funny wrong answers that earn bonus points. There's a genuine strategic tension: maximize correctness vs. maximize humor. The ideal prompt would need very precise topic boundaries.

## Smart Routing

**TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?**

With only 1 agent, all TF-IDF similarity scores were 0.000 — there was nothing to differentiate. With multiple agents, TF-IDF would match keyword overlap (e.g., "gold" and "chemical" would boost a science agent), but it would miss semantic meaning. A question about "the Iron Throne" wouldn't match a Movies agent unless its description contained those exact words.

Semantic embeddings (e.g., from OpenAI or sentence-transformers) would capture meaning: "Iron Throne" → "Game of Thrones" → "TV shows." This would dramatically improve routing accuracy.

Self-reported confidence is even more interesting: route the question to all agents, let each return a confidence score, and only use the highest-confidence answer. The cost is latency (you query everyone), but the quality improves. You could also do a two-phase approach: ask agents "can you answer this?" (cheap) before sending the full task (expensive).

## Trust and Reliability

**In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?**

Several mechanisms:
- **Timeouts:** The registry already uses 30-second timeouts for broadcasts. Slow agents get skipped.
- **Health checks:** The registry's background health checker marks agents offline after 3 consecutive failures. This is already implemented in `a2a_registry.py`.
- **Reputation/scoring:** Track answer quality over time. Agents that consistently return garbage get deprioritized or removed.
- **Redundancy:** Send the same question to multiple agents and cross-validate answers. If 3 out of 4 agree, use the consensus.
- **Circuit breakers:** After N failures, stop routing to an agent for a cooldown period rather than hammering a broken service.

## Scaling

**What would break if there were 1,000 agents instead of 20? What architectural changes would you need?**

What breaks at 1,000 agents:
- **Broadcast becomes O(N):** Sending every question to all 1,000 agents means 1,000 HTTP requests per question. A 24-question tournament = 24,000 requests. Latency would be terrible.
- **Registry as bottleneck:** A single-process FastAPI server with an in-memory dict doesn't scale horizontally.
- **ngrok free tier:** Each student needs their own tunnel. At scale, you'd need a proper deployment (Kubernetes, cloud functions).

Architectural changes needed:
- **Smart routing becomes mandatory** — you can't broadcast to 1,000 agents. TF-IDF or embedding-based routing to top-K agents is essential.
- **Async fan-out:** Use `asyncio.gather()` or a task queue (Celery, RabbitMQ) for parallel agent calls instead of sequential loops.
- **Persistent registry:** Move from in-memory dict to Redis or a database. Add replication.
- **Agent groups/namespaces:** Partition agents by capability domain so routing only searches relevant subsets.
