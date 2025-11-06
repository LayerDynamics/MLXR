#!/usr/bin/env python3
"""
Health and metrics example using MLXR Python SDK.

Demonstrates checking daemon health and retrieving metrics.
"""

from mlxrunner import MLXR

def main():
    client = MLXR()

    print("Health and Metrics Example")
    print("=" * 50)

    # Check health
    print("\n1. Daemon health:")
    try:
        health = client.health()
        print(f"Status: {health.get('status', 'unknown')}")
        if 'version' in health:
            print(f"Version: {health['version']}")
        if 'uptime' in health:
            print(f"Uptime: {health['uptime']}s")
    except Exception as e:
        print(f"Error: {e}")

    # Get metrics
    print("\n2. Daemon metrics:")
    try:
        metrics = client.metrics()

        # Display key metrics
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
    except Exception as e:
        print(f"Error: {e}")

    client.close()


if __name__ == "__main__":
    main()
