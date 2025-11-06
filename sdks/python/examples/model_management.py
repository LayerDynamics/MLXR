#!/usr/bin/env python3
"""
Model management example using MLXR Python SDK.

Demonstrates listing, pulling, and managing models.
"""

from mlxrunner import MLXR

def main():
    client = MLXR()

    print("Model Management Example")
    print("=" * 50)

    # List models (OpenAI API)
    print("\n1. List models (OpenAI API):")
    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id}")

    # List models (Ollama API)
    print("\n2. List models (Ollama API):")
    ollama_models = client.ollama.list()
    for model in ollama_models.models:
        print(f"  - {model.name} ({model.size} bytes)")

    # List running models
    print("\n3. List running models:")
    running = client.ollama.ps()
    if running.models:
        for model in running.models:
            print(f"  - {model.name} (expires: {model.expires_at})")
    else:
        print("  No models currently running")

    # Pull a model (streaming)
    print("\n4. Pull a model (example - commented out):")
    print("  # Uncomment to actually pull:")
    print("  # for chunk in client.ollama.pull('llama2:7b', stream=True):")
    print("  #     print(f'  Status: {chunk.status}')")

    # Show model info
    print("\n5. Show model info (example - commented out):")
    print("  # Uncomment if you have a model:")
    print("  # info = client.ollama.show('TinyLlama-1.1B')")
    print("  # print(f'  Family: {info.details.family}')")
    print("  # print(f'  Parameters: {info.details.parameter_size}')")

    client.close()


if __name__ == "__main__":
    main()
