#!/usr/bin/env python3
"""
Ollama generate example using MLXR Python SDK.

Demonstrates the Ollama-compatible generate API.
"""

from mlxrunner import MLXR

def main():
    client = MLXR()

    print("Ollama Generate Example")
    print("=" * 50)

    # Non-streaming generate
    print("\n1. Non-streaming generate:")
    response = client.ollama.generate(
        model="TinyLlama-1.1B",
        prompt="Why is the sky blue?",
        stream=False
    )
    print(f"Response: {response.response}")

    # Streaming generate
    print("\n2. Streaming generate:")
    print("Prompt: Write a haiku about computers\n")
    print("Response: ", end="", flush=True)

    for chunk in client.ollama.generate(
        model="TinyLlama-1.1B",
        prompt="Write a haiku about computers",
        stream=True
    ):
        if not chunk.done:
            print(chunk.response, end="", flush=True)

    print("\n")
    client.close()


if __name__ == "__main__":
    main()
