# task2_verbose_tracing.py
# Task 2: Modified langgraph_simple_agent.py with verbose/quiet tracing mode
# If input is "verbose" - enable tracing, if "quiet" - disable tracing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


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


class AgentState(TypedDict):
    """State object that flows through the LangGraph nodes."""
    user_input: str
    should_exit: bool
    llm_response: str
    verbose: bool  # NEW: Track verbose mode


def create_llm():
    """Create and configure the LLM using HuggingFace's transformers library."""
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
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
    print("Model loaded successfully!")
    return llm


def create_graph(llm):
    """Create the LangGraph state graph with verbose/quiet tracing."""

    def get_user_input(state: AgentState) -> dict:
        """Node that prompts the user for input via stdin."""
        verbose = state.get("verbose", False)
        
        if verbose:
            print("[TRACE] Entering get_user_input node")
            print(f"[TRACE] Current state: {state}")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("Commands: 'verbose' to enable tracing, 'quiet' to disable")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        # Handle verbose/quiet commands
        if user_input.lower() == "verbose":
            print("[TRACE] Verbose mode ENABLED")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": True
            }
        elif user_input.lower() == "quiet":
            print("Quiet mode enabled - tracing disabled")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": False
            }
        elif user_input.lower() in ['quit', 'exit', 'q']:
            if verbose:
                print("[TRACE] User requested exit")
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True
            }

        if verbose:
            print(f"[TRACE] Exiting get_user_input with input: '{user_input}'")

        return {
            "user_input": user_input,
            "should_exit": False
        }

    def call_llm(state: AgentState) -> dict:
        """Node that invokes the LLM with the user's input."""
        verbose = state.get("verbose", False)
        user_input = state["user_input"]

        if verbose:
            print("[TRACE] Entering call_llm node")
            print(f"[TRACE] User input: '{user_input}'")

        # Skip LLM call for verbose/quiet commands
        if user_input.lower() in ["verbose", "quiet"]:
            if verbose:
                print("[TRACE] Skipping LLM - mode change command")
            return {"llm_response": f"Mode set to: {user_input}"}

        prompt = f"User: {user_input}\nAssistant:"

        if verbose:
            print(f"[TRACE] Sending prompt to LLM: '{prompt}'")

        print("\nProcessing your input...")
        response = llm.invoke(prompt)

        if verbose:
            print(f"[TRACE] LLM response received, length: {len(response)}")

        return {"llm_response": response}

    def print_response(state: AgentState) -> dict:
        """Node that prints the LLM's response to stdout."""
        verbose = state.get("verbose", False)

        if verbose:
            print("[TRACE] Entering print_response node")

        # Don't print mode change messages as full responses
        if state["user_input"].lower() not in ["verbose", "quiet"]:
            print("\n" + "-" * 50)
            print("LLM Response:")
            print("-" * 50)
            print(state["llm_response"])

        if verbose:
            print("[TRACE] Exiting print_response node")

        return {}

    def route_after_input(state: AgentState) -> str:
        """Routing function that determines the next node based on state."""
        verbose = state.get("verbose", False)

        if verbose:
            print(f"[TRACE] Router: should_exit={state.get('should_exit', False)}")

        if state.get("should_exit", False):
            return END
        return "call_llm"

    # Build the graph
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {"call_llm": "call_llm", END: END}
    )
    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph_task2.png"):
    """Generate a Mermaid diagram of the graph and save it as PNG."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Task 2: LangGraph with Verbose/Quiet Tracing")
    print("=" * 50)
    print()

    llm = create_llm()
    print("\nCreating LangGraph...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "verbose": False  # Start in quiet mode
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
