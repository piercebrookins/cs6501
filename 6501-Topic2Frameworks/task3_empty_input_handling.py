# task3_empty_input_handling.py
# Task 3: Handle empty input with 3-way conditional branch
# Empty input loops back to get_user_input (never passed to LLM)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


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
    is_empty_input: bool  # NEW: Track if input was empty


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
    """
    Create the LangGraph state graph with 3-way conditional routing.
    
    Graph structure:
        START -> get_user_input -> [conditional] -> call_llm -> print_response -+
                       ^   ^              |                                      |
                       |   |              +-> END (if user wants to quit)        |
                       |   |              |                                      |
                       |   +--------------+ (if empty input - loop back)         |
                       |                                                         |
                       +---------------------------------------------------------+
    """

    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin.
        Sets is_empty_input flag to enable 3-way routing.
        """
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        # Check if user wants to exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "is_empty_input": False
            }

        # Check if input is empty (or whitespace only)
        if not user_input.strip():
            print("⚠️  Empty input detected! Please enter something.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "is_empty_input": True  # Signal empty input
            }

        # Valid non-empty input
        return {
            "user_input": user_input,
            "should_exit": False,
            "is_empty_input": False
        }

    def call_llm(state: AgentState) -> dict:
        """Node that invokes the LLM with the user's input."""
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        print("\nProcessing your input...")
        response = llm.invoke(prompt)

        return {"llm_response": response}

    def print_response(state: AgentState) -> dict:
        """Node that prints the LLM's response to stdout."""
        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])
        return {}

    def route_after_input(state: AgentState) -> Literal["call_llm", "get_user_input", "__end__"]:
        """
        3-way routing function that determines the next node based on state.
        
        Routes:
        1. should_exit=True -> END (user wants to quit)
        2. is_empty_input=True -> get_user_input (loop back for another try)
        3. Otherwise -> call_llm (process the input)
        """
        # Route 1: User wants to exit
        if state.get("should_exit", False):
            return END
        
        # Route 2: Empty input - loop back to get_user_input
        if state.get("is_empty_input", False):
            return "get_user_input"
        
        # Route 3: Valid input - proceed to LLM
        return "call_llm"

    # Build the graph
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    
    # 3-way conditional edges from get_user_input
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",         # Valid input -> LLM
            "get_user_input": "get_user_input",  # Empty input -> loop back
            END: END                          # Quit -> terminate
        }
    )
    
    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph_task3.png"):
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
    print("Task 3: Empty Input Handling with 3-Way Routing")
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
        "is_empty_input": False
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
