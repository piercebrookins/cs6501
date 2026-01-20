# task5_model_routing.py
# Task 5: Conditional routing based on "Hey Qwen" prefix
# If input starts with "Hey Qwen", route to Qwen, otherwise route to Llama

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
    use_qwen: bool  # NEW: Route to Qwen if True
    llm_response: str
    active_model: str  # Track which model responded


def create_llm(model_id: str, device: str):
    """Create and configure an LLM using HuggingFace's transformers library.
    
    Returns both the pipeline and tokenizer so we can use proper chat templates.
    """
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
        return_full_text=False,  # Only return the generated text, not the prompt
    )

    print(f"Model {model_id} loaded successfully!")
    return pipe, tokenizer  # Return both for proper chat formatting


def create_graph(llama_pipe, llama_tokenizer, qwen_pipe, qwen_tokenizer):
    """
    Create the LangGraph state graph with model-selective routing.
    
    Graph structure:
        START -> get_user_input -> [conditional] -+-> call_llama --+-> print_response -+
                       ^                 |        |                |                   |
                       |                 |        +-> call_qwen ---+                   |
                       |                 |                                             |
                       |                 +-> END (if quit)                             |
                       |                 +-> get_user_input (if empty)                 |
                       |                                                               |
                       +---------------------------------------------------------------+
    """

    def get_user_input(state: AgentState) -> dict:
        """Node that prompts the user for input via stdin."""
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("TIP: Start with 'qwen' or 'hey qwen' to use Qwen, otherwise Llama responds")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "is_empty_input": False,
                "use_qwen": False
            }

        if not user_input.strip():
            print("⚠️  Empty input detected! Please enter something.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "is_empty_input": True,
                "use_qwen": False
            }

        # Check if input mentions Qwen (case insensitive)
        # Triggers: "hey qwen", "qwen", "@qwen", starts with "qwen"
        input_lower = user_input.lower()
        use_qwen = (
            input_lower.startswith("hey qwen") or
            input_lower.startswith("qwen") or
            input_lower.startswith("@qwen") or
            "hey qwen" in input_lower
        )
        
        if use_qwen:
            print("🐦 Routing to Qwen...")
            # Strip common Qwen prefixes for cleaner input
            cleaned_input = user_input
            for prefix in ["hey qwen", "Hey Qwen", "HEY QWEN", "@qwen", "@Qwen", "qwen", "Qwen", "QWEN"]:
                if cleaned_input.startswith(prefix):
                    cleaned_input = cleaned_input[len(prefix):].strip()
                    break
            # Also strip leading punctuation like comma or colon
            if cleaned_input and cleaned_input[0] in ",.:!":
                cleaned_input = cleaned_input[1:].strip()
        else:
            print("🦙 Routing to Llama...")
            cleaned_input = user_input

        return {
            "user_input": cleaned_input if cleaned_input else user_input,
            "should_exit": False,
            "is_empty_input": False,
            "use_qwen": use_qwen
        }

    def call_llama(state: AgentState) -> dict:
        """Node that invokes Llama with the user's input using proper chat template."""
        user_input = state["user_input"]
        
        # Use proper chat template for Llama instruct models
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
            {"role": "user", "content": user_input}
        ]
        prompt = llama_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print("\n🦙 Llama is thinking...")
        result = llama_pipe(prompt)
        response = result[0]["generated_text"].strip()

        return {"llm_response": response, "active_model": "Llama"}

    def call_qwen(state: AgentState) -> dict:
        """Node that invokes Qwen with the user's input using proper chat template."""
        user_input = state["user_input"]
        
        # Use proper chat template for Qwen instruct models
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
            {"role": "user", "content": user_input}
        ]
        prompt = qwen_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print("\n🐦 Qwen is thinking...")
        result = qwen_pipe(prompt)
        response = result[0]["generated_text"].strip()

        return {"llm_response": response, "active_model": "Qwen"}

    def print_response(state: AgentState) -> dict:
        """Node that prints the LLM's response."""
        model = state.get("active_model", "Unknown")
        emoji = "🦙" if model == "Llama" else "🐦"
        
        print("\n" + "=" * 50)
        print(f"{emoji} {model.upper()} RESPONSE:")
        print("=" * 50)
        print(state.get("llm_response", "No response"))
        return {}

    def route_after_input(state: AgentState) -> Literal["call_llama", "call_qwen", "get_user_input", "__end__"]:
        """4-way routing function based on input."""
        if state.get("should_exit", False):
            return END
        if state.get("is_empty_input", False):
            return "get_user_input"
        if state.get("use_qwen", False):
            return "call_qwen"
        return "call_llama"

    # Build the graph
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
    
    # Both models converge to print_response
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph_task5.png"):
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
    print("Task 5: Model Routing (Hey Qwen -> Qwen, else Llama)")
    print("=" * 50)
    print()

    device = get_device()
    
    # Load both models (now returns pipe + tokenizer tuples)
    print("\n--- Loading Llama ---")
    llama_pipe, llama_tokenizer = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    
    print("\n--- Loading Qwen ---")
    qwen_pipe, qwen_tokenizer = create_llm("Qwen/Qwen2.5-0.5B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_pipe, llama_tokenizer, qwen_pipe, qwen_tokenizer)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "is_empty_input": False,
        "use_qwen": False,
        "llm_response": "",
        "active_model": ""
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
