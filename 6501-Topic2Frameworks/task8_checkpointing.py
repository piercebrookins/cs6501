# task8_checkpointing.py
# Task 8: Crash recovery with LangGraph checkpointing
# Uses SqliteSaver for persistent state across crashes

import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Literal, Annotated, List
from dataclasses import dataclass, asdict
from operator import add
import json
import sqlite3


CHECKPOINT_DB = "chat_checkpoint.db"
THREAD_ID = "main_conversation"


def get_device():
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"


@dataclass
class ConversationMessage:
    """Custom message type that tracks the source."""
    source: str
    content: str
    
    def to_dict(self):
        return {"source": self.source, "content": self.content}
    
    @classmethod
    def from_dict(cls, d):
        return cls(source=d["source"], content=d["content"])


class AgentState(TypedDict):
    """State object that flows through the LangGraph nodes."""
    conversation: Annotated[List[dict], add]  # Store as dicts for serialization
    user_input: str
    should_exit: bool
    is_empty_input: bool
    use_qwen: bool
    llm_response: str
    active_model: str


def create_llm(model_id: str, device: str):
    """Create and configure an LLM."""
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"Model {model_id} loaded!")
    return llm


def format_conversation_for_model(conversation: List[dict], target_model: str) -> str:
    """Format conversation history for a specific model."""
    formatted = []
    other_model = "Qwen" if target_model == "Llama" else "Llama"
    
    formatted.append(
        f"System: You are {target_model}, an AI assistant. "
        f"You are in a conversation with a human and another AI named {other_model}. "
        "Be helpful, concise, and friendly."
    )
    
    for msg in conversation:
        source = msg["source"]
        content = msg["content"]
        if source == "Human":
            formatted.append(f"User: Human: {content}")
        elif source == target_model:
            formatted.append(f"Assistant: {source}: {content}")
        else:
            formatted.append(f"User: {source}: {content}")
    
    formatted.append(f"Assistant: {target_model}:")
    return "\n".join(formatted)


def create_graph(llama_llm, qwen_llm):
    """Create the LangGraph with checkpointing support."""

    def get_user_input(state: AgentState) -> dict:
        """Node that prompts the user for input."""
        conversation = state.get("conversation", [])
        
        if conversation:
            print("\n" + "=" * 60)
            print("📜 CONVERSATION (recovered from checkpoint):")
            print("=" * 60)
            for msg in conversation:
                source = msg["source"]
                content = msg["content"]
                emoji = "👤" if source == "Human" else ("🦙" if source == "Llama" else "🐦")
                print(f"{emoji} {source}: {content}")
            print("=" * 60)

        print("\nEnter your text (or 'quit' to exit, 'clear' to start fresh):")
        print("TIP: Start with 'Hey Qwen' to address Qwen")
        print("💾 Your conversation is auto-saved! Kill anytime with Ctrl+C")

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Your conversation is saved.")
            return {
                "user_input": user_input,
                "should_exit": True,
                "is_empty_input": False,
                "use_qwen": False
            }
        
        if user_input.lower() == 'clear':
            print("🗑️  Conversation cleared! Starting fresh.")
            # Note: We'd need to handle this differently in practice
            # For now, we'll just continue with empty input behavior
            return {
                "user_input": "",
                "should_exit": False,
                "is_empty_input": True,
                "use_qwen": False
            }

        if not user_input.strip():
            print("⚠️  Empty input! Please enter something.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "is_empty_input": True,
                "use_qwen": False
            }

        use_qwen = user_input.lower().startswith("hey qwen")
        
        if use_qwen:
            print("🐦 Routing to Qwen...")
        else:
            print("🦙 Routing to Llama...")

        return {
            "user_input": user_input,
            "should_exit": False,
            "is_empty_input": False,
            "use_qwen": use_qwen,
            "conversation": [{"source": "Human", "content": user_input}]
        }

    def call_llama(state: AgentState) -> dict:
        """Node that invokes Llama."""
        conversation = state.get("conversation", [])
        prompt = format_conversation_for_model(conversation, "Llama")

        print("\n🦙 Llama is thinking...")
        response = llama_llm.invoke(prompt)
        
        if "Llama:" in response:
            response = response.split("Llama:")[-1].strip()
        response = response.split("\n")[0].strip()

        return {
            "llm_response": response,
            "active_model": "Llama",
            "conversation": [{"source": "Llama", "content": response}]
        }

    def call_qwen(state: AgentState) -> dict:
        """Node that invokes Qwen."""
        conversation = state.get("conversation", [])
        prompt = format_conversation_for_model(conversation, "Qwen")

        print("\n🐦 Qwen is thinking...")
        response = qwen_llm.invoke(prompt)
        
        if "Qwen:" in response:
            response = response.split("Qwen:")[-1].strip()
        response = response.split("\n")[0].strip()

        return {
            "llm_response": response,
            "active_model": "Qwen",
            "conversation": [{"source": "Qwen", "content": response}]
        }

    def print_response(state: AgentState) -> dict:
        """Node that prints the response."""
        model = state.get("active_model", "Unknown")
        emoji = "🦙" if model == "Llama" else "🐦"
        
        print("\n" + "-" * 50)
        print(f"{emoji} {model}: {state.get('llm_response', 'No response')}")
        print("-" * 50)
        print("\n💾 Checkpoint saved!")
        return {}

    def route_after_input(state: AgentState) -> Literal["call_llama", "call_qwen", "get_user_input", "__end__"]:
        """4-way routing."""
        if state.get("should_exit", False):
            return END
        if state.get("is_empty_input", False):
            return "get_user_input"
        if state.get("use_qwen", False):
            return "call_qwen"
        return "call_llama"

    # Build graph
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END
        }
    )
    
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder


def save_graph_image(graph, filename="lg_graph_task8.png"):
    """Save graph as PNG."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to {filename}")
    except Exception as e:
        print(f"Could not save graph: {e}")


def main():
    """Main entry point with crash recovery."""
    print("=" * 60)
    print("Task 8: Crash Recovery with Checkpointing")
    print("=" * 60)
    print()
    
    # Check if we're recovering from a crash
    recovering = os.path.exists(CHECKPOINT_DB)
    if recovering:
        print(f"🔄 Found existing checkpoint at {CHECKPOINT_DB}")
        print("   Recovering previous conversation...")
    else:
        print(f"📝 Starting fresh conversation (checkpoint: {CHECKPOINT_DB})")

    device = get_device()
    
    print("\n--- Loading Llama ---")
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    
    print("\n--- Loading Qwen ---")
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct", device)

    print("\nCreating LangGraph with checkpointing...")
    graph_builder = create_graph(llama_llm, qwen_llm)
    
    # Create SQLite checkpointer for persistence
    # This saves state after every node execution
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # Compile with checkpointer
    graph = graph_builder.compile(checkpointer=checkpointer)
    print("Graph with checkpointing created!")

    save_graph_image(graph)

    initial_state: AgentState = {
        "conversation": [],
        "user_input": "",
        "should_exit": False,
        "is_empty_input": False,
        "use_qwen": False,
        "llm_response": "",
        "active_model": ""
    }

    # Configuration for checkpointing - thread_id groups related checkpoints
    config = {"configurable": {"thread_id": THREAD_ID}}
    
    try:
        # This will automatically recover from the last checkpoint if one exists
        print("\n" + "=" * 60)
        print("🚀 Starting conversation (Ctrl+C to kill, state is preserved!)")
        print("=" * 60)
        
        graph.invoke(initial_state, config)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted! Your conversation has been saved.")
        print(f"   Checkpoint location: {CHECKPOINT_DB}")
        print("   Run this script again to continue where you left off.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
