# task7_chat_history_multimodel.py
# Task 7: Integrated chat history with Llama/Qwen model switching
# 
# Message formatting rules:
# - Human messages: {role: "user", content: "Human: <message>"}
# - Current model's messages: {role: "assistant", content: "<Model>: <message>"}
# - Other model's messages: {role: "user", content: "<Model>: <message>"}

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import TypedDict, Literal, Annotated, List
from dataclasses import dataclass
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


@dataclass
class ConversationMessage:
    """Custom message type that tracks the source (Human, Llama, or Qwen)."""
    source: str  # "Human", "Llama", or "Qwen"
    content: str


class AgentState(TypedDict):
    """State object that flows through the LangGraph nodes."""
    conversation: Annotated[List[ConversationMessage], add]  # Full conversation
    user_input: str
    should_exit: bool
    is_empty_input: bool
    use_qwen: bool
    llm_response: str
    active_model: str


def create_llm(model_id: str, device: str):
    """Create and configure an LLM using HuggingFace's transformers library."""
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


def format_conversation_for_model(
    conversation: List[ConversationMessage], 
    target_model: str
) -> str:
    """
    Format conversation history for a specific model.
    
    Rules per the assignment:
    - Human messages: "User: Human: <message>"
    - Target model's own messages: "Assistant: <Model>: <message>"
    - Other model's messages: "User: <Model>: <message>"
    """
    formatted = []
    other_model = "Qwen" if target_model == "Llama" else "Llama"
    
    # Add system prompt
    formatted.append(
        f"System: You are {target_model}, an AI assistant. "
        f"You are in a conversation with a human and another AI named {other_model}. "
        "Be helpful, concise, and friendly. You may agree or disagree with the other AI."
    )
    
    for msg in conversation:
        if msg.source == "Human":
            formatted.append(f"User: Human: {msg.content}")
        elif msg.source == target_model:
            # This model's own previous response
            formatted.append(f"Assistant: {msg.source}: {msg.content}")
        else:
            # Other model's response - treat as user input
            formatted.append(f"User: {msg.source}: {msg.content}")
    
    formatted.append(f"Assistant: {target_model}:")  # Prompt for response
    return "\n".join(formatted)


def create_graph(llama_llm, qwen_llm):
    """
    Create the LangGraph with chat history and model switching.
    """

    def get_user_input(state: AgentState) -> dict:
        """Node that prompts the user for input."""
        # Display conversation history
        conversation = state.get("conversation", [])
        if conversation:
            print("\n" + "=" * 60)
            print("📜 CONVERSATION:")
            print("=" * 60)
            for msg in conversation:
                emoji = "👤" if msg.source == "Human" else ("🦙" if msg.source == "Llama" else "🐦")
                print(f"{emoji} {msg.source}: {msg.content}")
            print("=" * 60)

        print("\nEnter your text (or 'quit' to exit):")
        print("TIP: Start with 'Hey Qwen' to address Qwen, otherwise Llama responds")

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
            print("⚠️  Empty input! Please enter something.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "is_empty_input": True,
                "use_qwen": False
            }

        # Check for "Hey Qwen" prefix
        use_qwen = user_input.lower().startswith("hey qwen")
        
        if use_qwen:
            print("🐦 Routing to Qwen...")
        else:
            print("🦙 Routing to Llama...")

        # Add human message to conversation
        return {
            "user_input": user_input,
            "should_exit": False,
            "is_empty_input": False,
            "use_qwen": use_qwen,
            "conversation": [ConversationMessage(source="Human", content=user_input)]
        }

    def call_llama(state: AgentState) -> dict:
        """Node that invokes Llama with formatted conversation."""
        conversation = state.get("conversation", [])
        prompt = format_conversation_for_model(conversation, "Llama")

        print("\n🦙 Llama is thinking...")
        response = llama_llm.invoke(prompt)
        
        # Clean response
        if "Llama:" in response:
            response = response.split("Llama:")[-1].strip()
        response = response.split("\n")[0].strip()  # Take first line only

        return {
            "llm_response": response,
            "active_model": "Llama",
            "conversation": [ConversationMessage(source="Llama", content=response)]
        }

    def call_qwen(state: AgentState) -> dict:
        """Node that invokes Qwen with formatted conversation."""
        conversation = state.get("conversation", [])
        prompt = format_conversation_for_model(conversation, "Qwen")

        print("\n🐦 Qwen is thinking...")
        response = qwen_llm.invoke(prompt)
        
        # Clean response
        if "Qwen:" in response:
            response = response.split("Qwen:")[-1].strip()
        response = response.split("\n")[0].strip()

        return {
            "llm_response": response,
            "active_model": "Qwen",
            "conversation": [ConversationMessage(source="Qwen", content=response)]
        }

    def print_response(state: AgentState) -> dict:
        """Node that prints the response."""
        model = state.get("active_model", "Unknown")
        emoji = "🦙" if model == "Llama" else "🐦"
        
        print("\n" + "-" * 50)
        print(f"{emoji} {model}: {state.get('llm_response', 'No response')}")
        print("-" * 50)
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

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph_task7.png"):
    """Save graph as PNG."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to {filename}")
    except Exception as e:
        print(f"Could not save graph: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Task 7: Multi-Model Chat with Shared History")
    print("=" * 60)
    print()

    device = get_device()
    
    print("\n--- Loading Llama ---")
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    
    print("\n--- Loading Qwen ---")
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created!")

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

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
