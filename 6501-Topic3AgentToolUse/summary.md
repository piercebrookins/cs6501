# 📚 Topic 3: Agent Tool Use - Summary

> **Course:** CS6501 - LLM Agents  
> **Focus:** How LLM agents leverage external tools to extend their capabilities

---

## 🎯 What I Learned

- How to run and benchmark local LLM servers with Ollama
- Integrating commercial LLMs (GPT-4o-mini) via the OpenAI API
- Understanding the OpenAI tool/function calling protocol
- Building and orchestrating tools with LangGraph
- Implementing persistent conversations with SQLite checkpointing

---

## ✅ Task Accomplishments

### Task 1: Ollama Server Performance Benchmarking

**What I Built:** Two MMLU evaluation scripts running against a local Ollama server with Llama 3.2-1B.

**What I Demonstrated:**

| Execution Mode | Time | Observation |
|----------------|------|-------------|
| **Sequential** | ~2m 58s | Programs run one after another |
| **Parallel** | ~2m 48s | Slight improvement (~10 sec saved) |

**Results:**
- **Astronomy:** 68/152 correct = **44.74%** accuracy (105.2 seconds)
- **Business Ethics:** 52/100 correct = **52.00%** accuracy (72.4 seconds)

**Key Finding:** Parallel execution provides minimal speedup because Ollama serializes GPU operations. The real win is **model loading happens once** at server startup rather than per-script.

---

### Task 2: OpenAI API Configuration

**What I Built:** A verification script to confirm OpenAI API connectivity.

**What I Demonstrated:**
```
Response from GPT-4o-mini: Working! What can I
OpenAI API is configured correctly!

Usage:
  Prompt tokens: 11
  Completion tokens: 5
  Total tokens: 16
```

**Accomplishment:** Successfully configured API key management and verified GPT-4o-mini responds correctly with minimal token usage.

---

### Task 3: Manual Tool Handling with Calculator

**What I Built:** A comprehensive calculator tool with 20+ operations (arithmetic, trigonometry, geometry) and manual tool dispatch logic.

**What I Demonstrated:**

| Query | Tool Call | Result |
|-------|-----------|--------|
| "15 multiplied by 7" | `multiply(15, 7)` | **105** |
| "Area of circle, radius 5" | `area_circle(5)` | **78.54 sq units** |
| "Sine of 45 degrees" | `sin(45)` | **0.8509** |
| "Volume of sphere, radius 3" | `volume_sphere(3)` | **113.10 cubic units** |
| "Hypotenuse with sides 3, 4" | `hypotenuse(3, 4)` | **5.0** |
| "Volume of cone, r=2, h=6" | `volume_cone(2, 6)` | **25.13 cubic units** |

**Accomplishment:** The LLM correctly identified which calculator operation to use for each natural language query and properly formatted the arguments.

---

### Task 4: LangGraph Multi-Tool Orchestration

**What I Built:** Three LangGraph tools using the `@tool` decorator:
1. **Calculator** - arithmetic & geometry
2. **Letter Counter** - case-insensitive character counting
3. **Word Analyzer** - text statistics (word count, character count, most common letter, etc.)

**What I Demonstrated:**

#### Single Tool Use
- "25 times 17?" → `calculator(multiply, 25, 17)` → **425** ✅

#### Parallel Tool Calls (Same Turn!)
- "More i's or s's in Mississippi riverboats?" 
  - Turn 1: `count_letter("i")` AND `count_letter("s")` called **in parallel**
  - Turn 2: `calculator(subtract, 5, 5)` → 0
  - Result: **Equal! Both have 5** ✅

#### Sequential Chaining (Multi-Turn)
- "Sine of the difference between i's and s's in Mississippi riverboats"
  - Turn 1: Count i's (5), count s's (5) - parallel
  - Turn 2: `subtract(5, 5)` → 0
  - Turn 3: `sin(0)` → **0.0** ✅

#### All Three Tools Combined
- "Analyze 'mathematical calculation' - word count, a's, sqrt of letter count"
  - Turn 1: `word_analyzer` + `count_letter("a")` - parallel
  - Turn 4: `sqrt(23)` → **4.80**
  - Result: 2 words, 5 a's, √23 ≈ 4.80 ✅

#### Hitting the 5-Turn Limit
- Complex chained query: count i's → count s's → subtract → abs → multiply by π → area_circle
  - Successfully **hit all 5 turns** with sequential tool chaining! ✅

**Accomplishment:** Demonstrated parallel tool calls, sequential chaining, multi-tool queries, and pushed the turn limit.

---

### Task 5: Conversation Agent with Checkpointing

**What I Built:** A LangGraph state machine with:
- Agent node (LLM calls)
- Tools node (tool execution)
- Conditional routing (should_continue)
- SQLite checkpointing for persistence

**What I Demonstrated:**

```
You: Hi! What's 15 times 23?
Assistant: 15 times 23 is 345.

You: Now add 50 to that result.
Assistant: Adding 50 to 345 gives you 395.  ← REMEMBERS CONTEXT!

You: How many i's are in Mississippi?
Assistant: There are 4 "i's" in "Mississippi."

You: What's the square root of that count?
Assistant: The square root of 4 is 2.  ← REMEMBERS PREVIOUS RESULT!

You: Can you remind me what calculation we did first?
Assistant: Sure! The first calculation we did was multiplying 15 by 23, which resulted in 345.  ← FULL MEMORY!
```

**Accomplishment:** 
- ✅ Conversation context persists across multiple turns
- ✅ Agent correctly references previous results ("that result", "that count")
- ✅ State checkpointed to `demo_checkpoint.db` - survives restarts!

---

### Task 6: Parallelization Analysis

**Question:** Where is there an opportunity for parallelization not being used?

**My Answer:** **Tool execution when multiple tools are called in the same turn.**

The current implementation processes tool calls sequentially:
```python
for tool_call in last_message.tool_calls:
    result = tool_map[tool_name].invoke(tool_args)  # Sequential!
```

Could be parallelized with:
```python
with ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda tc: tool_map[tc["name"]].invoke(tc["args"]), tool_calls))
```

**When it matters:** External API calls, database queries, I/O-bound operations.

---

## 📁 Portfolio Files

| File | Description |
|------|-------------|
| `task1_mmlu_astronomy.py` | MMLU eval on astronomy via Ollama |
| `task1_mmlu_business_ethics.py` | MMLU eval on business ethics via Ollama |
| `task2_openai_test.py` | OpenAI API verification |
| `task3_manual_tool_handling.py` | Manual tool dispatch with calculator |
| `task4_langgraph_tools.py` | LangGraph multi-tool agent |
| `task5_langgraph_conversation.py` | Stateful conversation with checkpointing |
| `output_task*.txt` | Execution logs demonstrating results |
| `demo_checkpoint.db` | SQLite database with persisted conversation |

---

## 🔑 Key Takeaways

1. **Local LLMs via Ollama** are easy to set up but GPU serialization limits parallel gains
2. **GPT-4o-mini** is excellent for tool use - fast, cheap, and reliable
3. **Manual tool handling** requires boilerplate but gives full control
4. **LangGraph `@tool`** decorator + tool maps = clean, extensible code
5. **Parallel tool calls** happen automatically when the LLM decides it's safe
6. **Checkpointing** enables persistent, resumable conversations across restarts
7. **Sequential chaining** can hit turn limits on complex queries

---

*Completed for CS6501 Topic 3: Agent Tool Use* 🐕
