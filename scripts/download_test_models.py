#!/usr/bin/env python3
"""
Download test models from HuggingFace for CI/testing.

Models are downloaded to test_models/ directory and stored in Git LFS
for reuse across workflow runs.

Usage:
    python scripts/download_test_models.py [--model MODEL_NAME] [--force]

Examples:
    # Download default models
    python scripts/download_test_models.py

    # Download specific model
    python scripts/download_test_models.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Force re-download even if cached
    python scripts/download_test_models.py --force
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not installed", file=sys.stderr)
    print("Install with: pip install huggingface-hub", file=sys.stderr)
    sys.exit(1)

# Test models configuration
# Small models suitable for CI testing
TEST_MODELS = {
    "tinyllama": {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "allow_patterns": [
            "*.json",
            "*.safetensors",
            "tokenizer.model",
            "*.txt",
        ],
        "ignore_patterns": [
            "*.bin",  # Prefer safetensors
            "*.gguf",  # Too large for test downloads
        ],
        "description": "TinyLlama 1.1B - Small model for basic testing",
        "size_mb": 4400,  # Approximate size
    },
    "tinyllama-gguf": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "allow_patterns": [
            "*.json",
            "*Q4_K_M.gguf",  # Download only Q4_K_M quant
            "tokenizer.model",
        ],
        "ignore_patterns": [],
        "description": "TinyLlama 1.1B Q4_K_M GGUF - Quantized model for testing",
        "size_mb": 669,  # Q4_K_M is ~669MB
    },
}

# Output directory
MODELS_DIR = Path("test_models")


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_model_files(model_dir: Path) -> Dict[str, str]:
    """Get list of model files with their SHA256 hashes."""
    files = {}
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(model_dir)
            files[str(rel_path)] = calculate_sha256(file_path)
    return files


def create_model_manifest(model_dir: Path, model_config: dict) -> None:
    """Create manifest file with model metadata."""
    manifest = {
        "repo_id": model_config["repo_id"],
        "description": model_config["description"],
        "size_mb": model_config["size_mb"],
        "files": get_model_files(model_dir),
    }

    manifest_path = model_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")


def download_model(
    model_name: str,
    model_config: dict,
    force: bool = False,
) -> bool:
    """
    Download a model from HuggingFace.

    Args:
        model_name: Short name for the model
        model_config: Model configuration dict
        force: Force re-download even if cached

    Returns:
        True if successful, False otherwise
    """
    repo_id = model_config["repo_id"]
    output_dir = MODELS_DIR / model_name

    # Check if already downloaded
    if output_dir.exists() and not force:
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            print(f"✓ Model '{model_name}' already downloaded to {output_dir}")
            print(f"  Use --force to re-download")
            return True

    print(f"Downloading model: {model_name}")
    print(f"  Repository: {repo_id}")
    print(f"  Output: {output_dir}")
    print(f"  Estimated size: {model_config['size_mb']} MB")

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download model files
        print(f"  Downloading files...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=model_config.get("allow_patterns"),
            ignore_patterns=model_config.get("ignore_patterns"),
        )

        # Create manifest
        create_model_manifest(output_dir, model_config)

        print(f"✓ Successfully downloaded '{model_name}'")
        return True

    except HfHubHTTPError as e:
        print(f"✗ HTTP error downloading '{model_name}': {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Error downloading '{model_name}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def list_models() -> None:
    """List available test models."""
    print("Available test models:")
    print()
    for name, config in TEST_MODELS.items():
        print(f"  {name}:")
        print(f"    Repository: {config['repo_id']}")
        print(f"    Description: {config['description']}")
        print(f"    Size: ~{config['size_mb']} MB")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download test models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        help="Specific model to download (default: all)",
        choices=list(TEST_MODELS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models if requested
    if args.list:
        list_models()
        return 0

    # Determine which models to download
    if args.model == "all":
        models_to_download = TEST_MODELS.items()
    else:
        models_to_download = [(args.model, TEST_MODELS[args.model])]

    # Download models
    print(f"Model download directory: {MODELS_DIR.absolute()}")
    print()

    success_count = 0
    total_count = len(models_to_download)

    for model_name, model_config in models_to_download:
        if download_model(model_name, model_config, force=args.force):
            success_count += 1
        print()

    # Summary
    print("=" * 60)
    print(f"Downloaded {success_count}/{total_count} models")

    if success_count < total_count:
        print("⚠ Some models failed to download")
        return 1
    else:
        print("✓ All models downloaded successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
