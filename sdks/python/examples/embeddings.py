#!/usr/bin/env python3
"""
Embeddings example using MLXR Python SDK.

Demonstrates generating embeddings for text.
"""

from mlxrunner import MLXR

def main():
    client = MLXR()

    print("Embeddings Example")
    print("=" * 50)

    # Generate embeddings
    print("\n1. Generate embedding for a single text:")
    response = client.embeddings.create(
        model="TinyLlama-1.1B",
        input="Hello, world!"
    )

    embedding = response.data[0].embedding
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Generate embeddings for multiple texts
    print("\n2. Generate embeddings for multiple texts:")
    response = client.embeddings.create(
        model="TinyLlama-1.1B",
        input=[
            "Machine learning is fascinating",
            "Python is a great language",
            "Apple Silicon is powerful"
        ]
    )

    print(f"Generated {len(response.data)} embeddings")
    for i, emb in enumerate(response.data):
        print(f"  Text {i+1}: {len(emb.embedding)} dimensions")

    # Using Ollama API
    print("\n3. Generate embedding with Ollama API:")
    ollama_response = client.ollama.embeddings(
        model="TinyLlama-1.1B",
        prompt="MLXR is fast"
    )
    print(f"Embedding dimension: {len(ollama_response.embedding)}")

    client.close()


if __name__ == "__main__":
    main()
