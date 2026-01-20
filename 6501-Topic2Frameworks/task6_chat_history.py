# task6_chat_history.py
# Task 6: Add chat history using LangChain Message API
# Uses HumanMessage, AIMessage, SystemMessage for conversation context

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Literal, Annotated
from operator import add


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
    messages: Annotated[list, add]  # Chat history accumulates
    user_input: str
    should_exit: bool
    is_empty_input: bool
    llm_response: str


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


def format_messages_for_llm(messages: list) -> str:
    """
    Convert LangChain messages to a prompt string for the LLM.
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    
    formatted.append("Assistant:")  # Prompt for next response
    return "\n".join(formatted)


def create_graph(llm):
    """
    Create the LangGraph state graph with chat history.
    
    The messages list accumulates throughout the conversation,
    providing context for each LLM call.
    """

    def get_user_input(state: AgentState) -> dict:
        """Node that prompts the user for input via stdin."""
        # Show conversation history
        if state.get("messages"):
            print("\n" + "-" * 50)
            print("📜 CONVERSATION HISTORY:")
            print("-" * 50)
            for msg in state["messages"]:
                if isinstance(msg, SystemMessage):
                    continue  # Don't show system messages
                role = "You" if isinstance(msg, HumanMessage) else "🦙 Llama"
                print(f"{role}: {msg.content}")
            print("-" * 50)

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("Chat history is maintained!")
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

        # Add user message to history
        return {
            "user_input": user_input,
            "should_exit": False,
            "is_empty_input": False,
            "messages": [HumanMessage(content=user_input)]
        }

    def call_llm(state: AgentState) -> dict:
        """Node that invokes the LLM with full conversation context."""
        messages = state.get("messages", [])
        
        # Format all messages into a prompt
        prompt = format_messages_for_llm(messages)

        print("\n🦙 Llama is thinking (with full context)...")
        response = llm.invoke(prompt)
        
        # Clean up response (remove the prompt echo if present)
        # Some models repeat the prompt, so we try to extract just the new response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Add AI response to history
        return {
            "llm_response": response,
            "messages": [AIMessage(content=response)]
        }

    def print_response(state: AgentState) -> dict:
        """Node that prints the LLM's response."""
        print("\n" + "=" * 50)
        print("🦙 LLAMA RESPONSE:")
        print("=" * 50)
        print(state.get("llm_response", "No response"))
        return {}

    def route_after_input(state: AgentState) -> Literal["call_llm", "get_user_input", "__end__"]:
        """3-way routing function."""
        if state.get("should_exit", False):
            return END
        if state.get("is_empty_input", False):
            return "get_user_input"
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
        {
            "call_llm": "call_llm",
            "get_user_input": "get_user_input",
            END: END
        }
    )
    
    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph_task6.png"):
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
    print("Task 6: Chat History with Message API")
    print("=" * 50)
    print()

    device = get_device()
    
    print("\n--- Loading Llama ---")
    llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Initialize with system message
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content="You are a helpful AI assistant named Llama. Be concise and friendly.")
        ],
        "user_input": "",
        "should_exit": False,
        "is_empty_input": False,
        "llm_response": ""
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
