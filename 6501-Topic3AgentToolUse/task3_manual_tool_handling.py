"""
Task 3: Manual Tool Handling with Calculator

This script demonstrates manual tool handling with GPT-4o-mini.
Includes a calculator tool with geometric functions.

Usage: python task3_manual_tool_handling.py
"""

import os
import json
import math
from openai import OpenAI

client = OpenAI()

# Define the calculator tool schema for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "A calculator that can perform arithmetic and geometric calculations. "
                          "Supports: add, subtract, multiply, divide, power, sqrt, "
                          "sin, cos, tan, asin, acos, atan, log, ln, exp, "
                          "area_circle, area_rectangle, area_triangle, "
                          "circumference, volume_sphere, volume_cylinder, volume_cone, "
                          "hypotenuse, degrees_to_radians, radians_to_degrees",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": [
                            "add", "subtract", "multiply", "divide", "power", "sqrt",
                            "sin", "cos", "tan", "asin", "acos", "atan",
                            "log", "ln", "exp",
                            "area_circle", "area_rectangle", "area_triangle",
                            "circumference", "volume_sphere", "volume_cylinder", "volume_cone",
                            "hypotenuse", "degrees_to_radians", "radians_to_degrees"
                        ]
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand (or radius/base/angle depending on operation)"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand (or height/width depending on operation). Optional for unary operations."
                    },
                    "c": {
                        "type": "number",
                        "description": "Third operand (for operations like volume_cylinder that need height). Optional."
                    }
                },
                "required": ["operation", "a"]
            }
        }
    }
]


def calculator(operation: str, a: float, b: float = None, c: float = None) -> str:
    """
    Calculator with arithmetic and geometric functions.

    Args:
        operation: The operation to perform
        a: First operand
        b: Second operand (optional for unary operations)
        c: Third operand (optional, for some volume calculations)

    Returns:
        JSON string with the result or error
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

        # Trigonometric functions (input in radians)
        elif operation == "sin":
            result = math.sin(a)
        elif operation == "cos":
            result = math.cos(a)
        elif operation == "tan":
            result = math.tan(a)
        elif operation == "asin":
            if a < -1 or a > 1:
                return json.dumps({"error": "asin input must be between -1 and 1"})
            result = math.asin(a)
        elif operation == "acos":
            if a < -1 or a > 1:
                return json.dumps({"error": "acos input must be between -1 and 1"})
            result = math.acos(a)
        elif operation == "atan":
            result = math.atan(a)

        # Logarithmic and exponential
        elif operation == "log":
            if a <= 0:
                return json.dumps({"error": "log input must be positive"})
            result = math.log10(a)
        elif operation == "ln":
            if a <= 0:
                return json.dumps({"error": "ln input must be positive"})
            result = math.log(a)
        elif operation == "exp":
            result = math.exp(a)

        # Geometric: Areas
        elif operation == "area_circle":
            # a = radius
            result = math.pi * a * a
        elif operation == "area_rectangle":
            # a = length, b = width
            result = a * b
        elif operation == "area_triangle":
            # a = base, b = height
            result = 0.5 * a * b

        # Geometric: Circumference
        elif operation == "circumference":
            # a = radius
            result = 2 * math.pi * a

        # Geometric: Volumes
        elif operation == "volume_sphere":
            # a = radius
            result = (4/3) * math.pi * (a ** 3)
        elif operation == "volume_cylinder":
            # a = radius, b = height
            result = math.pi * (a ** 2) * b
        elif operation == "volume_cone":
            # a = radius, b = height
            result = (1/3) * math.pi * (a ** 2) * b

        # Geometric: Hypotenuse
        elif operation == "hypotenuse":
            # a, b = sides of right triangle
            result = math.sqrt(a**2 + b**2)

        # Angle conversions
        elif operation == "degrees_to_radians":
            result = math.radians(a)
        elif operation == "radians_to_degrees":
            result = math.degrees(a)

        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})

        return json.dumps({"result": result, "operation": operation})

    except Exception as e:
        return json.dumps({"error": str(e)})


def run_conversation(user_message: str):
    """
    Run a conversation with the model, handling tool calls manually.
    """
    print(f"\n{'='*70}")
    print(f"User: {user_message}")
    print('='*70)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to a calculator tool. "
                      "IMPORTANT: You MUST use the calculator tool for ALL mathematical "
                      "calculations, including simple arithmetic. Never calculate in your head. "
                      "Always use the tool and report the exact result it returns."
        },
        {"role": "user", "content": user_message}
    ]

    # First API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # Check if the model wants to call a tool
    while assistant_message.tool_calls:
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n[Tool Call] {function_name}")
            print(f"  Arguments: {json.dumps(function_args, indent=2)}")

            if function_name == "calculator":
                result = calculator(**function_args)
            else:
                result = json.dumps({"error": f"Unknown function: {function_name}"})

            print(f"  Result: {result}")

            # Add the assistant message and tool result to messages
            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Get the next response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message

    # Final response
    print(f"\nAssistant: {assistant_message.content}")
    return assistant_message.content


def main():
    print("\n" + "="*70)
    print("Task 3: Manual Tool Handling with Calculator")
    print("="*70)

    # Test queries demonstrating different calculator functions
    test_queries = [
        # Basic arithmetic
        "What is 15 multiplied by 7?",

        # Geometric calculations
        "What is the area of a circle with radius 5?",

        # Trigonometry
        "What is the sine of 45 degrees?",

        # Combined calculations
        "Calculate the volume of a sphere with radius 3",

        # Multi-step calculation
        "If I have a right triangle with sides 3 and 4, what is the hypotenuse?",

        # Complex geometric problem
        "What is the volume of a cone with radius 2 and height 6?",
    ]

    for query in test_queries:
        run_conversation(query)
        print("\n")


if __name__ == "__main__":
    main()
