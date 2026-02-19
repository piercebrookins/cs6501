# Topic 4: Exploring Tools

Portfolio artifacts for CS6501 Topic 4.

## Table of Contents

- [Assignment Prompt](./topic4.md)
- [Task 1 Program: Manual ToolNode Agent](./task1_toolnode_manual_implementation.py)
- [Task 2 Program: ReAct Agent Wrapper](./task2_react_agent_wrapper.py)
- [Core Implementation: ToolNode](./toolnode_example.py)
- [Core Implementation: ReAct](./react_agent_example.py)
- [Terminal Output: Task 1](./output_task1_toolnode_terminal.txt)
- [Terminal Output: Task 2](./output_task2_react_agent_terminal.txt)
- [Written Answers to Portfolio Questions](./Topic4_Portfolio_Answers.md)
- [Generated Graph: Manual ToolNode](./langchain_manual_tool_graph.png)
- [Generated Graph: ReAct Internal](./langchain_react_agent.png)
- [Generated Graph: ReAct Conversation](./langchain_conversation_graph.png)
- [2-Hour Project Folder](./2HourProject/)
  - [Project PRD](./2HourProject/PRD-SmartTravelPlanner.md)
  - [Project README](./2HourProject/smart-travel-planner/README.md)

---

## What Was Completed

### 1) Study and run `toolnode_example.py` and `react_agent_example.py`
- ✅ Completed
- Implementations included in:
  - `toolnode_example.py`
  - `react_agent_example.py`
- Task-numbered wrappers included for portfolio naming clarity:
  - `task1_toolnode_manual_implementation.py`
  - `task2_react_agent_wrapper.py`

### 2) Compare Mermaid graph outputs
- ✅ Completed
- Graph artifacts:
  - `langchain_manual_tool_graph.png`
  - `langchain_react_agent.png`
  - `langchain_conversation_graph.png`

### 3) Answer required analysis questions
- ✅ Completed in:
  - `Topic4_Portfolio_Answers.md`

Covered questions:
- Python features enabling ToolNode parallel dispatch
- Which tools benefit most from parallel dispatch
- How `verbose` / `exit` are handled in both programs
- Graph structure comparison between ToolNode and ReAct
- Example where ReAct is too restrictive and ToolNode is preferred

### 4) Save terminal outputs from running programs
- ✅ Completed
- Output logs:
  - `output_task1_toolnode_terminal.txt`
  - `output_task2_react_agent_terminal.txt`

### 5) Implement a 2-hour project
- ✅ Completed
- Project folder:
  - `2HourProject/smart-travel-planner/`
- Includes modular source code, tests, config, and project README.

---

## Run Instructions

```bash
# Manual ToolNode version
python task1_toolnode_manual_implementation.py

# ReAct version
python task2_react_agent_wrapper.py
```

> Note: Set required API credentials in your environment (e.g., OpenAI key) before running.

---

## Notes on Design Quality

- DRY: task-numbered files are thin wrappers to avoid duplicating large implementations.
- Single Responsibility: core logic stays in the original implementation files.
- Traceability: each assignment deliverable maps to a concrete artifact in this directory.
