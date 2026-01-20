# Topic 2: Agent Orchestration Frameworks

> **Course:** 6501 - Applied Large Language Models  
> **Topic:** Agent Orchestration with LangGraph

## 📚 Table of Contents

| File | Task | Description |
|------|------|-------------|
| [`langgraph_simple_agent.py`](./langgraph_simple_agent.py) | Base | Original starter code - simple LangGraph agent with Llama-3.2-1B-Instruct |
| [`task2_verbose_tracing.py`](./task2_verbose_tracing.py) | Task 2 | Adds `verbose`/`quiet` tracing mode - type "verbose" to enable node tracing |
| [`task3_empty_input_handling.py`](./task3_empty_input_handling.py) | Task 3 | Handles empty input with 3-way conditional routing (loops back instead of sending to LLM) |
| [`task4_parallel_models.py`](./task4_parallel_models.py) | Task 4 | Parallel execution of both Llama AND Qwen models |
| [`task5_model_routing.py`](./task5_model_routing.py) | Task 5 | Conditional routing: "Hey Qwen" → Qwen, otherwise → Llama |
| [`task6_chat_history.py`](./task6_chat_history.py) | Task 6 | Chat history with LangChain Message API (HumanMessage, AIMessage, SystemMessage) |
| [`task7_chat_history_multimodel.py`](./task7_chat_history_multimodel.py) | Task 7 | Integrated chat history with Llama/Qwen switching and proper role formatting |
| [`task8_checkpointing.py`](./task8_checkpointing.py) | Task 8 | Crash recovery with SQLite checkpointing - kill anytime, resume where you left off |

## 🎯 Learning Goals

- Understand LangGraph concepts: **nodes**, **edges**, **conditional edges**, **routers**, **parallelism**, and **checkpoints**
- Build a LangGraph multi-agent chat system

## 🛠️ Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Examples

```bash
# Task 2: Verbose/Quiet tracing
python task2_verbose_tracing.py

# Task 3: Empty input handling with 3-way routing
python task3_empty_input_handling.py

# Task 4: Parallel Llama + Qwen
python task4_parallel_models.py

# Task 5: Model routing based on "Hey Qwen"
python task5_model_routing.py

# Task 6: Chat history with Message API
python task6_chat_history.py

# Task 7: Multi-model chat with shared history
python task7_chat_history_multimodel.py

# Task 8: Crash recovery with checkpointing
python task8_checkpointing.py
```

## 📊 Graph Visualizations

 Each task generates a corresponding graph PNG:
- `lg_graph.png` - Original graph
- `lg_graph_task2.png` through `lg_graph_task8.png`

## 🔑 Key Concepts Demonstrated

### Nodes
Functions that process state and return updates:
```python
def get_user_input(state: AgentState) -> dict:
    # ... process input ...
    return {"user_input": input_text, "should_exit": False}
```

### Conditional Edges
Routing logic based on state:
```python
graph_builder.add_conditional_edges(
    "get_user_input",
    route_after_input,  # Routing function
    {"call_llm": "call_llm", "get_user_input": "get_user_input", END: END}
)
```

### Parallel Execution (Task 4)
Multiple edges from one node fan out to parallel execution:
```python
graph_builder.add_edge("fan_out", "call_llama")
graph_builder.add_edge("fan_out", "call_qwen")
```

### Checkpointing (Task 8)
Persistent state for crash recovery:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=checkpointer)
graph.invoke(initial_state, {"configurable": {"thread_id": "my_thread"}})
```

## 📝 Notes

- **Models Used:**
  - Llama: `meta-llama/Llama-3.2-1B-Instruct`
  - Qwen: `Qwen/Qwen2.5-0.5B-Instruct`
  
- **Device Support:** CUDA > MPS (Apple Silicon) > CPU (auto-detected)

- **Empty Input Behavior (Task 3 findings):** Small LLMs like Llama-3.2-1B can produce unpredictable or repetitive outputs when given empty prompts. The 3-way routing pattern prevents this by looping back to the input node.

## 📂 Terminal Output Files

(Add your terminal session outputs here as you run each task)

- `output_task2.txt` - Verbose/quiet tracing session
- `output_task3.txt` - Empty input handling session
- `output_task4.txt` - Parallel models session
- `output_task5.txt` - Model routing session
- `output_task6.txt` - Chat history session
- `output_task7.txt` - Multi-model chat session (interesting conversations)
- `output_task8.txt` - Checkpointing/crash recovery session

## 📚 Resources

- [LangChain Graph API Overview](https://python.langchain.com/docs/langgraph)
- [LangGraph Crash Recovery](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
