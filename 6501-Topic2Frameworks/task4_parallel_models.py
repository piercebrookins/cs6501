# task4_parallel_models.py
# Task 4: Parallel execution with both Llama and Qwen models
# Both models run in parallel and results are combined

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
    is_empty_input: bool
    llama_response: str  # Response from Llama
    qwen_response: str   # Response from Qwen


def create_llm(model_id: str, device: str):
    """Create and configure an LLM using HuggingFace's transformers library."""
    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

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
    print(f"Model {model_id} loaded successfully!")
    return llm


def create_graph(llama_llm, qwen_llm):
    """
    Create the LangGraph state graph with parallel model execution.
    
    Graph structure:
        START -> get_user_input -> [conditional] -> fan_out -+-> call_llama --+
                       ^                 |                   |                |
                       |                 |                   +-> call_qwen ---+-> combine_results -> print_response -+
                       |                 |                                                                           |
                       |                 +-> END (if user wants to quit)                                             |
                       |                 |                                                                           |
                       |                 +-> get_user_input (if empty)                                               |
                       |                                                                                             |
                       +---------------------------------------------------------------------------------------------+
    """

    def get_user_input(state: AgentState) -> dict:
        """Node that prompts the user for input via stdin."""
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("Both Llama AND Qwen will respond in parallel!")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "is_empty_input": False
            }

        if not user_input.strip():
            print("⚠️  Empty input detected! Please enter something.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "is_empty_input": True
            }

        return {
            "user_input": user_input,
            "should_exit": False,
            "is_empty_input": False
        }

    def fan_out(state: AgentState) -> dict:
        """Node that prepares state for parallel execution."""
        print("\n🚀 Sending to both Llama and Qwen in parallel...")
        return {}

    def call_llama(state: AgentState) -> dict:
        """Node that invokes Llama with the user's input."""
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        print("\n🦙 Llama is thinking...")
        response = llama_llm.invoke(prompt)

        return {"llama_response": response}

    def call_qwen(state: AgentState) -> dict:
        """Node that invokes Qwen with the user's input."""
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        print("\n🐦 Qwen is thinking...")
        response = qwen_llm.invoke(prompt)

        return {"qwen_response": response}

    def print_response(state: AgentState) -> dict:
        """Node that prints both models' responses."""
        print("\n" + "=" * 50)
        print("🦙 LLAMA RESPONSE:")
        print("=" * 50)
        print(state.get("llama_response", "No response"))

        print("\n" + "=" * 50)
        print("🐦 QWEN RESPONSE:")
        print("=" * 50)
        print(state.get("qwen_response", "No response"))
        return {}

    def route_after_input(state: AgentState) -> Literal["fan_out", "get_user_input", "__end__"]:
        """3-way routing function."""
        if state.get("should_exit", False):
            return END
        if state.get("is_empty_input", False):
            return "get_user_input"
        return "fan_out"

    # Build the graph
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("fan_out", fan_out)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "fan_out": "fan_out",
            "get_user_input": "get_user_input",
            END: END
        }
    )
    
    # Parallel edges from fan_out to both models
    graph_builder.add_edge("fan_out", "call_llama")
    graph_builder.add_edge("fan_out", "call_qwen")
    
    # Both converge to print_response
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph_task4.png"):
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
    print("Task 4: Parallel Llama + Qwen Execution")
    print("=" * 50)
    print()

    device = get_device()
    
    # Load both models
    print("\n--- Loading Llama ---")
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    
    print("\n--- Loading Qwen ---")
    # Using Qwen2.5-0.5B-Instruct as a lightweight alternative
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "is_empty_input": False,
        "llama_response": "",
        "qwen_response": ""
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
