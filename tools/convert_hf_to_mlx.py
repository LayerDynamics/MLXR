#!/usr/bin/env python3
"""
Convert HuggingFace model to MLX format
Supports: safetensors, pytorch checkpoint
Output: MLX npz format with model weights and config
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    from transformers import AutoConfig, AutoTokenizer
    import numpy as np
    from safetensors import safe_open
    import torch
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("\nInstall dependencies with:")
    print("  pip install mlx transformers safetensors torch")
    sys.exit(1)


def load_hf_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """Load weights from HuggingFace model (safetensors or pytorch)."""
    print(f"Loading weights from {model_path}")

    weights = {}

    # Try safetensors first
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        print(f"Found {len(safetensors_files)} safetensors files")
        for st_file in safetensors_files:
            with safe_open(st_file, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        return weights

    # Try pytorch checkpoint
    pt_files = list(model_path.glob("pytorch_model*.bin"))
    if pt_files:
        print(f"Found {len(pt_files)} pytorch checkpoint files")
        for pt_file in pt_files:
            checkpoint = torch.load(pt_file, map_location="cpu")
            for key, tensor in checkpoint.items():
                weights[key] = tensor.numpy()
        return weights

    raise ValueError(f"No model weights found in {model_path}")


def convert_weight_names_to_mlx(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert HuggingFace weight names to MLX naming convention."""
    mlx_weights = {}

    for key, value in weights.items():
        # Convert HF naming to MLX naming
        new_key = key

        # Layer normalization
        new_key = new_key.replace("layer_norm", "norm")
        new_key = new_key.replace("layernorm", "norm")

        # Attention layers
        new_key = new_key.replace("self_attn", "attention")
        new_key = new_key.replace("q_proj", "query_proj")
        new_key = new_key.replace("k_proj", "key_proj")
        new_key = new_key.replace("v_proj", "value_proj")
        new_key = new_key.replace("o_proj", "out_proj")

        # MLP layers
        new_key = new_key.replace("mlp.gate_proj", "mlp.gate")
        new_key = new_key.replace("mlp.up_proj", "mlp.up")
        new_key = new_key.replace("mlp.down_proj", "mlp.down")

        # Embeddings
        new_key = new_key.replace("embed_tokens", "embedding")
        new_key = new_key.replace("lm_head", "output")

        mlx_weights[new_key] = value

    return mlx_weights


def save_mlx_model(output_path: Path, weights: Dict[str, np.ndarray], config: Dict[str, Any]):
    """Save model in MLX format."""
    print(f"Saving MLX model to {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as NPZ
    weights_path = output_path / "weights.npz"
    np.savez(weights_path, **weights)
    print(f"  Saved weights: {weights_path}")

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config: {config_path}")

    # Calculate total size
    total_size = sum(w.nbytes for w in weights.values())
    total_params = sum(w.size for w in weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size: {total_size / 1024**3:.2f} GB")


def convert_config(hf_config: Any) -> Dict[str, Any]:
    """Convert HuggingFace config to MLX format."""
    config = {
        "model_type": hf_config.model_type,
        "vocab_size": hf_config.vocab_size,
        "hidden_size": hf_config.hidden_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "intermediate_size": getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4),
        "max_position_embeddings": hf_config.max_position_embeddings,
        "rms_norm_eps": getattr(hf_config, "rms_norm_eps", 1e-5),
    }

    # GQA support
    if hasattr(hf_config, "num_key_value_heads"):
        config["num_key_value_heads"] = hf_config.num_key_value_heads
    else:
        config["num_key_value_heads"] = config["num_attention_heads"]

    # RoPE
    if hasattr(hf_config, "rope_theta"):
        config["rope_theta"] = hf_config.rope_theta

    # Attention bias
    if hasattr(hf_config, "attention_bias"):
        config["attention_bias"] = hf_config.attention_bias

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Llama model
  python convert_hf_to_mlx.py --input meta-llama/Llama-2-7b-hf --output models/llama-2-7b-mlx

  # Convert with specific dtype
  python convert_hf_to_mlx.py --input mistralai/Mistral-7B-v0.1 --output models/mistral-7b --dtype float16
""",
    )

    parser.add_argument("--input", type=str, required=True, help="HuggingFace model path or repo ID")
    parser.add_argument("--output", type=str, required=True, help="Output directory for MLX model")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"],
                       help="Output dtype (default: float16)")
    parser.add_argument("--upload-repo", type=str, help="HuggingFace repo to upload converted model")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("HuggingFace → MLX Converter")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"DType:  {args.dtype}")
    print()

    # Load HuggingFace config
    try:
        hf_config = AutoConfig.from_pretrained(str(input_path))
        print(f"Loaded config: {hf_config.model_type}")
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1

    # Convert config
    mlx_config = convert_config(hf_config)

    # Load weights
    try:
        weights = load_hf_weights(input_path)
        print(f"Loaded {len(weights)} weight tensors")
    except Exception as e:
        print(f"ERROR: Failed to load weights: {e}")
        return 1

    # Convert weight names
    mlx_weights = convert_weight_names_to_mlx(weights)
    print(f"Converted weight names")

    # Convert dtype if needed
    if args.dtype != "float32":
        print(f"Converting to {args.dtype}...")
        for key in mlx_weights:
            if mlx_weights[key].dtype == np.float32:
                if args.dtype == "float16":
                    mlx_weights[key] = mlx_weights[key].astype(np.float16)
                elif args.dtype == "bfloat16":
                    mlx_weights[key] = mlx_weights[key].astype(np.uint16)  # BF16 as uint16

    # Save MLX model
    try:
        save_mlx_model(output_path, mlx_weights, mlx_config)
    except Exception as e:
        print(f"ERROR: Failed to save model: {e}")
        return 1

    # Copy tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(input_path))
        tokenizer.save_pretrained(output_path)
        print(f"  Saved tokenizer")
    except Exception as e:
        print(f"WARNING: Failed to save tokenizer: {e}")

    print()
    print("✓ Conversion complete!")

    if args.upload_repo:
        print(f"\nTo upload to HuggingFace Hub:")
        print(f"  huggingface-cli upload {args.upload_repo} {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
