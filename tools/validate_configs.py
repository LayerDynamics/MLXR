#!/usr/bin/env python3
"""
Configuration validation utility for MLXR
Validates server.yaml and model config files
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List


def validate_server_config(config_path: Path) -> bool:
    """Validate server configuration file."""
    print(f"Validating server config: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        errors = []
        warnings = []

        # Check required sections
        required_sections = ['transport', 'scheduler', 'kv_cache']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing '{section}' section")

        # Validate transport section
        if 'transport' in config:
            transport = config['transport']
            if 'uds' not in transport:
                warnings.append("Missing transport.uds section")
            elif 'path' not in transport['uds']:
                warnings.append("Missing transport.uds.path")

            # Check gRPC config if enabled
            if 'grpc' in transport:
                grpc = transport['grpc']
                if grpc.get('enabled', False):
                    if 'port' not in grpc:
                        warnings.append("gRPC enabled but no port specified")
                    if 'host' not in grpc:
                        warnings.append("gRPC enabled but no host specified")

        # Validate scheduler section
        if 'scheduler' in config:
            scheduler = config['scheduler']
            if 'max_batch_tokens' not in scheduler:
                warnings.append("Missing scheduler.max_batch_tokens")
            if 'max_batch_size' not in scheduler:
                warnings.append("Missing scheduler.max_batch_size")
            if 'target_latency_ms' not in scheduler:
                warnings.append("Missing scheduler.target_latency_ms")

        # Validate KV cache section
        if 'kv_cache' in config:
            kv = config['kv_cache']
            if 'block_size' not in kv:
                warnings.append("Missing kv_cache.block_size")
            if 'max_gpu_blocks' not in kv:
                warnings.append("Missing kv_cache.max_gpu_blocks")
            if 'max_cpu_blocks' not in kv:
                warnings.append("Missing kv_cache.max_cpu_blocks")

        # Validate speculative decoding section
        if 'speculative_decoding' in config:
            spec = config['speculative_decoding']
            if spec.get('enabled', False):
                if spec.get('draft_model') is None:
                    warnings.append("Speculative decoding enabled but draft_model is null (will auto-select)")

        # Report results
        for warning in warnings:
            print(f"  WARNING: {warning}")

        if errors:
            for error in errors:
                print(f"  ERROR: {error}")
            return False

        print("  ✓ Server config is valid")
        return True

    except yaml.YAMLError as e:
        print(f"  ERROR: YAML parse error: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def validate_model_config(config_path: Path) -> bool:
    """Validate model configuration file."""
    print(f"Validating model config: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        errors = []
        warnings = []

        # Check required sections
        if 'model' not in config:
            errors.append("Missing 'model' section")
            return False

        model = config['model']

        # Required model fields
        required_fields = ['name', 'family', 'path', 'format']
        for field in required_fields:
            if field not in model:
                errors.append(f"Missing model.{field}")

        # Validate format
        if 'format' in model:
            valid_formats = ['gguf', 'safetensors', 'mlx']
            if model['format'] not in valid_formats:
                warnings.append(f"Unknown format: {model['format']} (expected: {', '.join(valid_formats)})")

        # Check architecture section
        if 'architecture' not in config:
            warnings.append("Missing 'architecture' section")
        else:
            arch = config['architecture']
            arch_fields = [
                'vocab_size', 'hidden_size', 'num_hidden_layers',
                'num_attention_heads', 'intermediate_size'
            ]
            for field in arch_fields:
                if field not in arch:
                    warnings.append(f"Missing architecture.{field}")

            # Validate GQA if present
            if 'num_key_value_heads' in arch:
                num_kv_heads = arch['num_key_value_heads']
                num_q_heads = arch.get('num_attention_heads', 0)
                if num_kv_heads > num_q_heads:
                    errors.append(f"Invalid GQA: num_key_value_heads ({num_kv_heads}) > num_attention_heads ({num_q_heads})")

        # Check tokenizer section
        if 'tokenizer' not in config:
            warnings.append("Missing 'tokenizer' section")
        else:
            tokenizer = config['tokenizer']
            if 'type' not in tokenizer:
                warnings.append("Missing tokenizer.type")
            if 'path' not in tokenizer:
                warnings.append("Missing tokenizer.path")

        # Check context section
        if 'context' not in config:
            warnings.append("Missing 'context' section")

        # Report results
        for warning in warnings:
            print(f"  WARNING: {warning}")

        if errors:
            for error in errors:
                print(f"  ERROR: {error}")
            return False

        print("  ✓ Model config is valid")
        return True

    except yaml.YAMLError as e:
        print(f"  ERROR: YAML parse error: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Main entry point."""
    print("MLXR Configuration Validator\n")

    config_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "configs")

    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}")
        return 1

    errors = 0

    # Validate server config
    server_config = config_dir / "server.yaml"
    if server_config.exists():
        if not validate_server_config(server_config):
            errors += 1
    else:
        print(f"ERROR: Server config not found: {server_config}")
        errors += 1

    print()

    # Validate model configs
    models_dir = config_dir / "models"
    if models_dir.exists() and models_dir.is_dir():
        model_configs = sorted(models_dir.glob("*.yaml"))
        if not model_configs:
            print(f"WARNING: No model configs found in {models_dir}")
        else:
            for model_config in model_configs:
                if not validate_model_config(model_config):
                    errors += 1
                print()
    else:
        print(f"WARNING: Models directory not found: {models_dir}\n")

    if errors == 0:
        print("✓ All configurations are valid!")
        return 0
    else:
        print(f"✗ Found {errors} configuration error(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
