"""
Task 5: LangGraph Conversation Agent with Checkpointing

This script implements a long-running conversation agent using LangGraph
nodes and edges (instead of Python loops), with checkpointing and recovery.

Features:
- LangGraph state graph with nodes and edges
- Persistent conversation memory via SQLite checkpointing
- Tool use (calculator, letter counter, word analyzer)
- Recovery from previous sessions

Usage:
    python task5_langgraph_conversation.py

To continue a previous conversation, use the same thread_id.
"""

import os
import json
import math
import uuid
from typing import Annotated, TypedDict, Literal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================================
# TOOLS (same as Task 4)
# ============================================================================

@tool
def calculator(
    operation: Annotated[str, "The operation to perform"],
    a: Annotated[float, "First operand"],
    b: Annotated[float, "Second operand (optional)"] = None,
    c: Annotated[float, "Third operand (optional)"] = None
) -> str:
    """
    Calculator with arithmetic and geometric functions.
    Supports: add, subtract, multiply, divide, power, sqrt,
    sin, cos, tan, log, ln, exp, area_circle, volume_sphere,
    hypotenuse, degrees_to_radians, radians_to_degrees, abs, mod
    """
    try:
        result = None

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b if b != 0 else "Division by zero"
        elif operation == "power":
            result = a ** b
        elif operation == "sqrt":
            result = math.sqrt(a)
        elif operation == "abs":
            result = abs(a)
        elif operation == "mod":
            result = a % b
        elif operation == "sin":
            result = math.sin(a)
        elif operation == "cos":
            result = math.cos(a)
        elif operation == "tan":
            result = math.tan(a)
        elif operation == "log":
            result = math.log10(a)
        elif operation == "ln":
            result = math.log(a)
        elif operation == "exp":
            result = math.exp(a)
        elif operation == "area_circle":
            result = math.pi * a * a
        elif operation == "volume_sphere":
            result = (4/3) * math.pi * (a ** 3)
        elif operation == "hypotenuse":
            result = math.sqrt(a**2 + b**2)
        elif operation == "degrees_to_radians":
            result = math.radians(a)
        elif operation == "radians_to_degrees":
            result = math.degrees(a)
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})

        return json.dumps({"result": result, "operation": operation})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def count_letter(text: Annotated[str, "The text to search in"], letter: Annotated[str, "The letter to count"]) -> str:
    """Count occurrences of a letter in text (case-insensitive)."""
    count = text.lower().count(letter.lower())
    return json.dumps({"text": text, "letter": letter, "count": count})


@tool
def word_analyzer(text: Annotated[str, "The text to analyze"]) -> str:
    """Analyze text for word count, character count, and statistics."""
    words = text.split()
    letters_only = ''.join(c.lower() for c in text if c.isalpha())
    letter_freq = {}
    for char in letters_only:
        letter_freq[char] = letter_freq.get(char, 0) + 1
    most_common = max(letter_freq, key=letter_freq.get) if letter_freq else None

    return json.dumps({
        "word_count": len(words),
        "character_count": len(text),
        "most_common_letter": most_common
    })


# Tool list and map
tools = [calculator, count_letter, word_analyzer]
tool_map = {t.name: t for t in tools}

# Bind tools to the model
model_with_tools = model.bind_tools(tools)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for the conversation agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


# ============================================================================
# GRAPH NODES
# ============================================================================

def call_model(state: AgentState) -> dict:
    """Node: Call the LLM with the current messages."""
    messages = state["messages"]

    # Add system message if this is the first call
    system_message = {
        "role": "system",
        "content": """You are a helpful assistant with access to calculator, letter counting, and word analysis tools.

RULES:
1. ALWAYS use tools for calculations and counting - never do math in your head.
2. Use count_letter to count letters, calculator for math, word_analyzer for text stats.
3. Be conversational and remember the context of our conversation."""
    }

    # Call the model
    response = model_with_tools.invoke([system_message] + messages)

    return {"messages": [response], "turn_count": state.get("turn_count", 0)}


def call_tools(state: AgentState) -> dict:
    """Node: Execute tool calls from the last message."""
    last_message = state["messages"][-1]
    tool_messages = []

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  [Tool] {tool_name}: {json.dumps(tool_args)}")

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})

            print(f"  [Result] {result}")

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    new_turn_count = state.get("turn_count", 0) + 1
    return {"messages": tool_messages, "turn_count": new_turn_count}


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]

    # Check turn limit
    if state.get("turn_count", 0) >= 5:
        print("  [Reached turn limit]")
        return "end"

    # Check if there are tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    return "end"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_graph():
    """Build the LangGraph state graph."""
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    return workflow


# ============================================================================
# MAIN CONVERSATION LOOP
# ============================================================================

def print_mermaid_diagram():
    """Print the Mermaid diagram of the graph."""
    diagram = """
```mermaid
graph TD
    A[__start__] --> B[agent]
    B --> C{should_continue}
    C -->|tools| D[tools]
    C -->|end| E[__end__]
    D --> B
```
"""
    print(diagram)


def run_conversation():
    """Run an interactive conversation with checkpointing."""
    print("\n" + "="*70)
    print("Task 5: LangGraph Conversation Agent with Checkpointing")
    print("="*70)
    print("\nGraph Structure (Mermaid):")
    print_mermaid_diagram()

    # Build the graph
    workflow = build_graph()

    # Set up checkpointing with SQLite
    db_path = "conversation_checkpoint.db"
    print(f"\nCheckpoint database: {db_path}")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        # Compile the graph with checkpointer
        app = workflow.compile(checkpointer=checkpointer)

        # Get or create thread ID for this conversation
        thread_id = input("\nEnter thread ID (or press Enter for new conversation): ").strip()
        if not thread_id:
            thread_id = str(uuid.uuid4())[:8]
            print(f"Created new thread: {thread_id}")
        else:
            print(f"Resuming thread: {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}

        # Check for existing conversation
        state = app.get_state(config)
        if state.values and state.values.get("messages"):
            print("\n--- Previous conversation history ---")
            for msg in state.values["messages"]:
                if isinstance(msg, HumanMessage):
                    print(f"You: {msg.content}")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"Assistant: {msg.content}")
            print("--- End of history ---\n")

        print("\nCommands: 'quit' to exit, 'history' to show conversation")
        print("Tools available: calculator, count_letter, word_analyzer")
        print("-"*70)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("Goodbye! Your conversation has been saved.")
                    break

                if user_input.lower() == 'history':
                    state = app.get_state(config)
                    if state.values and state.values.get("messages"):
                        print("\n--- Conversation History ---")
                        for msg in state.values["messages"]:
                            if isinstance(msg, HumanMessage):
                                print(f"You: {msg.content}")
                            elif isinstance(msg, AIMessage) and msg.content:
                                print(f"Assistant: {msg.content}")
                        print("--- End ---")
                    continue

                # Process the message through the graph
                print("\n[Processing...]")
                inputs = {
                    "messages": [HumanMessage(content=user_input)],
                    "turn_count": 0
                }

                # Stream the response
                final_response = None
                for event in app.stream(inputs, config, stream_mode="values"):
                    messages = event.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                                final_response = last_msg.content

                if final_response:
                    print(f"\nAssistant: {final_response}")

            except KeyboardInterrupt:
                print("\n\nConversation saved. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def demo_mode():
    """Run a demo with preset queries to show functionality."""
    print("\n" + "="*70)
    print("Task 5: DEMO MODE - Showing tool use and conversation memory")
    print("="*70)

    workflow = build_graph()
    db_path = "demo_checkpoint.db"

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        thread_id = "demo-" + str(uuid.uuid4())[:4]
        config = {"configurable": {"thread_id": thread_id}}

        demo_queries = [
            "Hi! What's 15 times 23?",
            "Now add 50 to that result.",
            "How many i's are in Mississippi?",
            "What's the square root of that count?",
            "Thanks! Can you remind me what calculation we did first?",
        ]

        for query in demo_queries:
            print(f"\n{'='*50}")
            print(f"You: {query}")
            print("-"*50)

            inputs = {"messages": [HumanMessage(content=query)], "turn_count": 0}

            final_response = None
            for event in app.stream(inputs, config, stream_mode="values"):
                messages = event.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                            final_response = last_msg.content

            if final_response:
                print(f"\nAssistant: {final_response}")

    print("\n" + "="*70)
    print("Demo complete! The conversation was checkpointed to demo_checkpoint.db")
    print("="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        run_conversation()
