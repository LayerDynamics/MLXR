#!/usr/bin/env python3
"""
Basic chat example using MLXR Python SDK.

Demonstrates simple chat completion with the OpenAI-compatible API.
"""

from mlxrunner import MLXR

def main():
    # Initialize client
    client = MLXR()

    # Basic chat completion
    print("Basic Chat Example")
    print("=" * 50)

    response = client.chat.completions.create(
        model="TinyLlama-1.1B",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ],
        temperature=0.7,
        max_tokens=100
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens if response.usage else 'N/A'}")

    client.close()


if __name__ == "__main__":
    main()
