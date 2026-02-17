from __future__ import annotations

from typing import TypedDict

import gradio as gr
from langgraph.graph import END, START, StateGraph

from topic6vlm.config import VLMConfig
from topic6vlm.image_utils import resize_image_in_place
from topic6vlm.llava_client import LlavaClient


class ChatState(TypedDict):
    image_path: str
    user_message: str
    history: list[tuple[str, str]]
    assistant_reply: str


config = VLMConfig()
client = LlavaClient(model_name=config.model_name)


def _build_prompt(user_message: str, history: list[tuple[str, str]]) -> str:
    recent_turns = history[-6:]
    transcript = "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in recent_turns
    )
    return (
        "You are a helpful vision-language assistant. "
        "Use the provided image and conversation context. "
        "If unsure, say what is uncertain instead of guessing.\n\n"
        f"Conversation so far:\n{transcript}\n\n"
        f"User: {user_message}\nAssistant:"
    )


def assistant_turn(state: ChatState) -> ChatState:
    prompt = _build_prompt(state["user_message"], state["history"])
    response = client.chat(prompt=prompt, image_paths=[state["image_path"]])
    state["assistant_reply"] = response
    return state


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("assistant_turn", assistant_turn)
    graph.add_edge(START, "assistant_turn")
    graph.add_edge("assistant_turn", END)
    return graph.compile()


chat_graph = build_graph()


def _render_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "_No messages yet._"

    lines: list[str] = []
    for user_msg, assistant_msg in history:
        lines.append(f"**You:** {user_msg}")
        lines.append(f"**Assistant:** {assistant_msg}")
        lines.append("")
    return "\n".join(lines)


def chat_with_image(
    user_message: str,
    image_path: str | None,
    history: list[tuple[str, str]] | None,
) -> tuple[list[tuple[str, str]], str, str]:
    history = history or []
    if not image_path:
        return history, _render_history(history), "Upload an image first, then ask a question."

    optimized_image = resize_image_in_place(
        image_path=image_path,
        max_width=config.image_max_width,
        quality=config.image_quality,
    )

    result = chat_graph.invoke(
        {
            "image_path": optimized_image,
            "user_message": user_message,
            "history": history,
            "assistant_reply": "",
        }
    )
    assistant_reply = result["assistant_reply"]
    history.append((user_message, assistant_reply))
    return history, _render_history(history), ""


def launch_app() -> None:
    with gr.Blocks(title="Exercise 1: LLaVA LangGraph Chat") as demo:
        gr.Markdown("# Exercise 1: Vision-Language LangGraph Chat Agent")
        gr.Markdown(
            "Upload one image and ask multiple questions about it. "
            "The app keeps short conversation context in a LangGraph state flow."
        )

        image_input = gr.Image(type="filepath", label="Upload Image")
        history_state = gr.State([])
        conversation = gr.Markdown("_No messages yet._", label="Conversation")
        user_input = gr.Textbox(label="Your Question", lines=2)
        status = gr.Markdown("")

        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")

        send_btn.click(
            fn=chat_with_image,
            inputs=[user_input, image_input, history_state],
            outputs=[history_state, conversation, status],
        ).then(lambda: "", None, user_input)

        clear_btn.click(
            lambda: ([], "_No messages yet._", "", ""),
            None,
            [history_state, conversation, user_input, status],
        )

    demo.launch()


if __name__ == "__main__":
    launch_app()
