"""
Task 2: OpenAI GPT-4o Mini Test

This script verifies that the OpenAI API is set up correctly.

Setup:
1. Set environment variable: export OPENAI_API_KEY="your-key"
2. Install openai: pip install openai
3. Run: python task2_openai_test.py

For Google Colab:
    from google.colab import userdata
    import os
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
"""

import os
from openai import OpenAI

def main():
    # Create an OpenAI client instance
    # client = OpenAI() automatically reads the API key from the
    # OPENAI_API_KEY environment variable. It creates a connection
    # to OpenAI's API servers that can be reused for multiple requests.
    client = OpenAI()

    # client.chat.completions.create() sends a chat completion request
    # to the OpenAI API. It takes:
    # - model: which model to use (gpt-4o-mini is fast and cost-effective)
    # - messages: a list of message dictionaries with "role" and "content"
    #   - "role" can be "system", "user", or "assistant"
    #   - "content" is the text of the message
    # - max_tokens: limits the response length (5 tokens is very short)
    # The method returns a response object containing the model's reply.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say: Working!"}],
        max_tokens=5
    )

    # Extract and print the response
    message = response.choices[0].message.content
    print(f"Response from GPT-4o-mini: {message}")
    print("\nOpenAI API is configured correctly!")

    # Print usage info
    print(f"\nUsage:")
    print(f"  Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
