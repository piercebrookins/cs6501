#!/usr/bin/env python3
"""
Enhanced Chat Agent with Context Management

Features:
- Configurable context management (truncation, sliding window, summarization)
- --no-history flag to disable conversation memory
- Multiple model support
- Pickle-based session persistence

Usage:
    python enhanced_chat_agent.py [--no-history] [--model MODEL] [--context-strategy STRATEGY]

Context Strategies:
    - none: Let context grow without limit (original behavior)
    - truncate: Keep only last N messages
    - sliding: Sliding window with system prompt preserved
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen": "Qwen/Qwen2-0.5B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."
SESSION_FILE = "chat_session.pkl"

# Context management settings
MAX_HISTORY_MESSAGES = 10  # For truncation strategy
MAX_CONTEXT_TOKENS = 2048  # Approximate max tokens to keep


# ============================================================================
# CONTEXT MANAGEMENT
# ============================================================================

class ContextManager:
    """Manages conversation context with different strategies."""

    def __init__(self, strategy: str = "none", max_messages: int = MAX_HISTORY_MESSAGES):
        self.strategy = strategy
        self.max_messages = max_messages

    def manage(self, history: list[dict]) -> list[dict]:
        """Apply context management strategy to history."""
        if self.strategy == "none" or len(history) <= 2:
            return history

        if self.strategy == "truncate":
            return self._truncate(history)
        elif self.strategy == "sliding":
            return self._sliding_window(history)
        else:
            return history

    def _truncate(self, history: list[dict]) -> list[dict]:
        """Keep system prompt + last N messages."""
        # Separate system prompt from conversation
        system_msgs = [m for m in history if m["role"] == "system"]
        conv_msgs = [m for m in history if m["role"] != "system"]

        # Keep only last N conversation messages
        if len(conv_msgs) > self.max_messages:
            conv_msgs = conv_msgs[-self.max_messages :]

        return system_msgs + conv_msgs

    def _sliding_window(self, history: list[dict]) -> list[dict]:
        """Sliding window: preserve system + recent context."""
        # Same as truncate but could be extended for summarization
        return self._truncate(history)


# ============================================================================
# CHAT AGENT
# ============================================================================

class ChatAgent:
    """Enhanced chat agent with context management."""

    def __init__(
        self,
        model_name: str = "llama",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        use_history: bool = True,
        context_strategy: str = "sliding",
    ):
        self.model_name = model_name
        self.model_path = MODELS.get(model_name, model_name)
        self.system_prompt = system_prompt
        self.use_history = use_history
        self.context_manager = ContextManager(context_strategy)

        # Initialize history with system prompt
        self.history = [{"role": "system", "content": system_prompt}]

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"\n🔄 Loading model: {self.model_name}...")
        print(f"   Path: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
        )

        if device in ["mps", "cpu"]:
            self.model = self.model.to(device)

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"✓ Model loaded on {self.device}")

    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Add user message
        if self.use_history:
            self.history.append({"role": "user", "content": user_input})
            # Apply context management
            managed_history = self.context_manager.manage(self.history)
        else:
            # No history mode: only system + current user message
            managed_history = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ]

        # Tokenize conversation
        input_ids = self.tokenizer.apply_chat_template(
            managed_history,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        attention_mask = torch.ones_like(input_ids)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        new_tokens = outputs[0][input_ids.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Add to history if using history mode
        if self.use_history:
            self.history.append({"role": "assistant", "content": response})

        return response

    def get_history_stats(self) -> dict:
        """Get statistics about current conversation."""
        user_msgs = sum(1 for m in self.history if m["role"] == "user")
        asst_msgs = sum(1 for m in self.history if m["role"] == "assistant")
        return {
            "total_messages": len(self.history),
            "user_messages": user_msgs,
            "assistant_messages": asst_msgs,
        }

    def clear_history(self):
        """Clear conversation history but keep system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        print("🗑️  History cleared!")

    def save_session(self, filepath: str = SESSION_FILE):
        """Save conversation to file."""
        session_data = {
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "history": self.history,
            "use_history": self.use_history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(session_data, f)
        print(f"💾 Session saved to {filepath}")

    @classmethod
    def load_session(cls, filepath: str = SESSION_FILE) -> "ChatAgent":
        """Load conversation from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            model_name=data["model_name"],
            system_prompt=data["system_prompt"],
            use_history=data["use_history"],
        )
        agent.history = data["history"]
        print(f"📂 Session loaded from {filepath}")
        return agent


# ============================================================================
# MAIN
# ============================================================================

def print_help():
    """Print in-chat help."""
    print(
        """
╔══════════════════════════════════════════════════════════╗
║                    CHAT COMMANDS                         ║
╠══════════════════════════════════════════════════════════╣
║  /help     - Show this help message                      ║
║  /clear    - Clear conversation history                  ║
║  /stats    - Show conversation statistics                ║
║  /save     - Save session to file                        ║
║  /history  - Show current history                        ║
║  /quit     - Exit the chat                               ║
╚══════════════════════════════════════════════════════════╝
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Enhanced Chat Agent")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable conversation history (stateless mode)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="llama",
        help="Model to use (default: llama)",
    )
    parser.add_argument(
        "--context-strategy",
        choices=["none", "truncate", "sliding"],
        default="sliding",
        help="Context management strategy (default: sliding)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved session",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🤖 Enhanced Chat Agent")
    print("=" * 60)

    # Load or create agent
    if args.resume and Path(SESSION_FILE).exists():
        agent = ChatAgent.load_session()
    else:
        agent = ChatAgent(
            model_name=args.model,
            use_history=not args.no_history,
            context_strategy=args.context_strategy,
        )

    # Display settings
    history_mode = "ENABLED" if agent.use_history else "DISABLED"
    print(f"\n📝 History: {history_mode}")
    print(f"📊 Context Strategy: {args.context_strategy}")
    print("\nType /help for commands, /quit to exit.")
    print("=" * 60 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ["/quit", "/exit", "/q"]:
                agent.save_session()
                print("\n👋 Goodbye!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/clear":
                agent.clear_history()
            elif cmd == "/stats":
                stats = agent.get_history_stats()
                print(f"\n📊 Stats: {stats}\n")
            elif cmd == "/save":
                agent.save_session()
            elif cmd == "/history":
                print("\n📜 Current History:")
                for i, msg in enumerate(agent.history):
                    role = msg["role"].upper()
                    content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                    print(f"  [{i}] {role}: {content}")
                print()
            else:
                print(f"❓ Unknown command: {cmd}")
            continue

        # Get response
        print("Assistant: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
