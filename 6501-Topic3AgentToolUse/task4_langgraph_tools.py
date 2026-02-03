"""
Task 4: LangGraph Tool Handling

This script demonstrates LangGraph tool handling with multiple tools:
1. Calculator (with geometric functions)
2. Letter counter
3. Word analyzer (custom tool)

Demonstrates multiple tool use, sequential chaining, and tool_map dispatch.

Usage: python task4_langgraph_tools.py
"""

import os
import json
import math
from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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
    sin, cos, tan, asin, acos, atan, log, ln, exp,
    area_circle, area_rectangle, area_triangle,
    circumference, volume_sphere, volume_cylinder, volume_cone,
    hypotenuse, degrees_to_radians, radians_to_degrees, abs, mod
    """
    try:
        result = None

        # Basic arithmetic
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return json.dumps({"error": "Division by zero"})
            result = a / b
        elif operation == "power":
            result = a ** b
        elif operation == "sqrt":
            if a < 0:
                return json.dumps({"error": "Cannot take square root of negative number"})
            result = math.sqrt(a)
        elif operation == "abs":
            result = abs(a)
        elif operation == "mod":
            result = a % b

        # Trigonometric functions (input in radians)
        elif operation == "sin":
            result = math.sin(a)
        elif operation == "cos":
            result = math.cos(a)
        elif operation == "tan":
            result = math.tan(a)
        elif operation == "asin":
            result = math.asin(a)
        elif operation == "acos":
            result = math.acos(a)
        elif operation == "atan":
            result = math.atan(a)

        # Logarithmic and exponential
        elif operation == "log":
            result = math.log10(a)
        elif operation == "ln":
            result = math.log(a)
        elif operation == "exp":
            result = math.exp(a)

        # Geometric: Areas
        elif operation == "area_circle":
            result = math.pi * a * a
        elif operation == "area_rectangle":
            result = a * b
        elif operation == "area_triangle":
            result = 0.5 * a * b

        # Geometric: Circumference
        elif operation == "circumference":
            result = 2 * math.pi * a

        # Geometric: Volumes
        elif operation == "volume_sphere":
            result = (4/3) * math.pi * (a ** 3)
        elif operation == "volume_cylinder":
            result = math.pi * (a ** 2) * b
        elif operation == "volume_cone":
            result = (1/3) * math.pi * (a ** 2) * b

        # Geometric: Hypotenuse
        elif operation == "hypotenuse":
            result = math.sqrt(a**2 + b**2)

        # Angle conversions
        elif operation == "degrees_to_radians":
            result = math.radians(a)
        elif operation == "radians_to_degrees":
            result = math.degrees(a)

        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})

        return json.dumps({"result": result, "operation": operation, "inputs": {"a": a, "b": b, "c": c}})

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def count_letter(text: Annotated[str, "The text to search in"], letter: Annotated[str, "The letter to count"]) -> str:
    """
    Count the number of occurrences of a specific letter in a piece of text.
    Case-insensitive counting.
    """
    if len(letter) != 1:
        return json.dumps({"error": "Please provide exactly one letter to count"})

    count = text.lower().count(letter.lower())
    return json.dumps({
        "text": text,
        "letter": letter,
        "count": count
    })


@tool
def word_analyzer(text: Annotated[str, "The text to analyze"]) -> str:
    """
    Analyze text for various statistics:
    - Word count
    - Character count (with and without spaces)
    - Sentence count
    - Average word length
    - Longest word
    - Shortest word
    - Most common letter
    """
    # Word analysis
    words = text.split()
    word_count = len(words)

    # Character counts
    char_count_with_spaces = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))

    # Sentence count (approximate)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    if sentence_count == 0:
        sentence_count = 1  # Assume at least one sentence

    # Word length statistics
    if words:
        word_lengths = [len(word.strip('.,!?;:')) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths)
        longest_word = max(words, key=lambda w: len(w.strip('.,!?;:')))
        shortest_word = min(words, key=lambda w: len(w.strip('.,!?;:')))
    else:
        avg_word_length = 0
        longest_word = ""
        shortest_word = ""

    # Most common letter
    letters_only = ''.join(c.lower() for c in text if c.isalpha())
    if letters_only:
        letter_freq = {}
        for char in letters_only:
            letter_freq[char] = letter_freq.get(char, 0) + 1
        most_common_letter = max(letter_freq, key=letter_freq.get)
        most_common_count = letter_freq[most_common_letter]
    else:
        most_common_letter = None
        most_common_count = 0

    return json.dumps({
        "word_count": word_count,
        "character_count_with_spaces": char_count_with_spaces,
        "character_count_no_spaces": char_count_no_spaces,
        "sentence_count": sentence_count,
        "average_word_length": round(avg_word_length, 2),
        "longest_word": longest_word,
        "shortest_word": shortest_word,
        "most_common_letter": most_common_letter,
        "most_common_letter_count": most_common_count
    })


# Create tool list and tool_map for dispatch
tools = [calculator, count_letter, word_analyzer]
tool_map = {tool.name: tool for tool in tools}

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful assistant with access to calculator, letter counting, and word analysis tools.

IMPORTANT RULES:
1. ALWAYS use the calculator tool for ANY mathematical calculation, no matter how simple.
2. ALWAYS use the count_letter tool to count letters in text - never count manually.
3. ALWAYS use the word_analyzer tool when asked about text statistics.
4. When comparing counts, use count_letter for each letter, then use calculator to find the difference.
5. Never perform calculations or counting in your head - always use the tools."""

# Create the agent with tools
agent = create_react_agent(
    model,
    tools,
    prompt=SYSTEM_PROMPT
)


def run_agent(query: str, max_turns: int = 5):
    """
    Run the agent with a query, demonstrating tool use.
    """
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)

    # Track turns for demonstration
    turn_count = 0
    inputs = {"messages": [("user", query)]}

    for event in agent.stream(inputs, stream_mode="values"):
        messages = event.get("messages", [])
        if messages:
            last_message = messages[-1]

            # Check if it's a tool call
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"\n[Turn {turn_count + 1}] Tool Call: {tool_call['name']}")
                    print(f"  Args: {json.dumps(tool_call['args'], indent=2)}")

            # Check if it's a tool response
            elif hasattr(last_message, 'type') and last_message.type == "tool":
                result = last_message.content
                print(f"  Result: {result}")
                turn_count += 1

            # Check if it's the final response
            elif hasattr(last_message, 'content') and last_message.content:
                if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                    print(f"\n[Final Response]")
                    print(f"Assistant: {last_message.content}")

        if turn_count >= max_turns:
            print(f"\n[Reached max turns limit: {max_turns}]")
            break

    return turn_count


def main():
    print("\n" + "="*70)
    print("Task 4: LangGraph Tool Handling with Multiple Tools")
    print("="*70)
    print("\nTools available:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # Test queries demonstrating various capabilities
    test_queries = [
        # Single tool use
        "What is 25 times 17?",

        # Letter counting
        "How many s's are in Mississippi riverboats?",

        # Word analysis
        "Analyze this text: The quick brown fox jumps over the lazy dog.",

        # Multiple tool use in one question (should trigger count_letter twice)
        "Are there more i's than s's in Mississippi riverboats?",

        # Sequential chaining - count letters then calculate
        "What is the difference between the number of i's and s's in Mississippi?",

        # Complex multi-tool query
        "What is the sine of the difference between the number of i's and the number of s's in Mississippi riverboats?",

        # Use all three tools
        "Analyze the phrase 'mathematical calculation' - tell me the word count, how many a's it has, and what is the square root of the letter count?",

        # Try to hit the turn limit with sequential operations
        "Count the i's in Mississippi, count the s's, subtract them, take the absolute value, multiply by pi, and then calculate the area of a circle with that radius.",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Test {i}")
        print(f"{'#'*70}")
        turns = run_agent(query)
        print(f"\n[Completed in {turns} tool turns]")


if __name__ == "__main__":
    main()
