#!/usr/bin/env python3
"""
Async chat example using MLXR Python SDK.

Demonstrates async/await usage with the async client.
"""

import asyncio
from mlxrunner import AsyncMLXR


async def main():
    async with AsyncMLXR() as client:
        print("Async Chat Example")
        print("=" * 50)

        # Async chat completion
        print("\n1. Async chat completion:")
        response = await client.chat.completions.create(
            model="TinyLlama-1.1B",
            messages=[
                {"role": "user", "content": "What is Python?"}
            ]
        )
        print(f"Response: {response.choices[0].message.content}")

        # Async streaming
        print("\n2. Async streaming:")
        print("Question: Count to 5\n")
        print("Response: ", end="", flush=True)

        stream = await client.chat.completions.create(
            model="TinyLlama-1.1B",
            messages=[
                {"role": "user", "content": "Count to 5"}
            ],
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
