#!/usr/bin/env python3
"""
Streaming chat example using MLXR Python SDK.

Demonstrates SSE streaming for real-time token generation.
"""

from mlxrunner import MLXR

def main():
    client = MLXR()

    print("Streaming Chat Example")
    print("=" * 50)
    print("Question: Tell me a short story about a robot.\n")
    print("Response: ", end="", flush=True)

    # Streaming chat completion
    stream = client.chat.completions.create(
        model="TinyLlama-1.1B",
        messages=[
            {"role": "user", "content": "Tell me a short story about a robot."}
        ],
        stream=True,
        max_tokens=200
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")
    client.close()


if __name__ == "__main__":
    main()
