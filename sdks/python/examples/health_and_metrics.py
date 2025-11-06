#!/usr/bin/env python3
"""
Health and metrics example using MLXR Python SDK.

Demonstrates checking daemon health and retrieving metrics.
"""

from typing import Dict, Any
from mlxrunner import MLXR


def display_health(health: Dict[str, Any]) -> None:
    """Display daemon health information."""
    print(f"Status: {health.get('status', 'unknown')}")
    if 'version' in health:
        print(f"Version: {health['version']}")
    if 'uptime' in health:
        print(f"Uptime: {health['uptime']}s")


def display_metrics(metrics: Dict[str, Any]) -> None:
    """Display daemon metrics."""
    if 'requests_total' in metrics:
        print(f"Total requests: {metrics['requests_total']}")
    if 'requests_active' in metrics:
        print(f"Active requests: {metrics['requests_active']}")
    if 'tokens_generated' in metrics:
        print(f"Tokens generated: {metrics['tokens_generated']}")
    if 'throughput_tokens_per_sec' in metrics:
        print(f"Throughput: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
    if 'latency_p95_ms' in metrics:
        print(f"P95 latency: {metrics['latency_p95_ms']:.2f}ms")
    if 'kv_cache_hit_rate' in metrics:
        print(f"KV cache hit rate: {metrics['kv_cache_hit_rate']:.2%}")


def main() -> None:
    """Main function to demonstrate health and metrics."""
    client = MLXR()

    print("Health and Metrics Example")
    print("=" * 50)

    # Check health
    print("\n1. Daemon health:")
    try:
        health = client.health()
        display_health(health)
    except Exception as e:
        print(f"Error: {e}")

    # Get metrics
    print("\n2. Daemon metrics:")
    try:
        metrics = client.metrics()
        display_metrics(metrics)
    except Exception as e:
        print(f"Error: {e}")

    client.close()


if __name__ == "__main__":
    main()
