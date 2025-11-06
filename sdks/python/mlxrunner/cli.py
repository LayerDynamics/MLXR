"""
Command-line interface for MLXR.

Provides CLI commands for model management, inference, and server operations.
"""

import functools
import sys
from typing import Callable, Optional

import click

from . import __version__
from .client import MLXR
from .exceptions import MLXRConnectionError, MLXRError


def with_client(f: Callable) -> Callable:
    """Decorator to inject MLXR client and handle errors/cleanup."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        base_url = ctx.obj.get("base_url")
        api_key = ctx.obj.get("api_key")

        try:
            client = MLXR(base_url=base_url, api_key=api_key)
        except MLXRConnectionError:
            click.echo("Error: Cannot connect to MLXR daemon.", err=True)
            click.echo("Is the daemon running? Try: mlxrunnerd", err=True)
            sys.exit(1)

        try:
            # Pass client as first argument after ctx
            return f(*args, client=client, **kwargs)
        except MLXRError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            client.close()

    return wrapper


@click.group()
@click.version_option(version=__version__)
@click.option("--base-url", help="HTTP base URL (e.g., http://localhost:11434)")
@click.option("--api-key", help="API key for authentication")
@click.pass_context
def main(ctx: click.Context, base_url: Optional[str], api_key: Optional[str]) -> None:
    """MLXR - High-performance LLM inference for Apple Silicon."""
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url
    ctx.obj["api_key"] = api_key


@main.command()
@click.pass_context
@with_client
def status(ctx: click.Context, client: MLXR) -> None:
    """Show MLXR daemon status and health."""
    health = client.health()
    click.echo("MLXR Daemon Status")
    click.echo("=" * 50)
    click.echo(f"Status: {health.get('status', 'unknown')}")

    if "version" in health:
        click.echo(f"Daemon version: {health['version']}")
    if "uptime" in health:
        uptime_s = health["uptime"]
        uptime_h = uptime_s // 3600
        uptime_m = (uptime_s % 3600) // 60
        click.echo(f"Uptime: {uptime_h}h {uptime_m}m")

    click.echo(f"\nSDK version: {__version__}")

    # Try to get metrics
    try:
        metrics = client.metrics()
        click.echo("\nMetrics:")
        if "requests_total" in metrics:
            click.echo(f"  Total requests: {metrics['requests_total']}")
        if "requests_active" in metrics:
            click.echo(f"  Active requests: {metrics['requests_active']}")
        if "throughput_tokens_per_sec" in metrics:
            click.echo(f"  Throughput: {metrics['throughput_tokens_per_sec']:.2f} tok/s")
    except Exception:
        pass  # Metrics endpoint might not be available


@main.group(name="models")
def models_group() -> None:
    """Model management commands."""
    pass


@models_group.command(name="list")
@click.option("--ollama", is_flag=True, help="Use Ollama API format")
@click.pass_context
@with_client
def list_models(ctx: click.Context, client: MLXR, ollama: bool) -> None:
    """List available models."""
    if ollama:
        models = client.ollama.list()
        click.echo("Available Models (Ollama format):")
        click.echo("=" * 50)
        for model in models.models:
            size_mb = model.size / (1024 * 1024)
            click.echo(f"  {model.name}")
            click.echo(f"    Size: {size_mb:.1f} MB")
            click.echo(f"    Modified: {model.modified_at}")
            click.echo()
    else:
        models = client.models.list()
        click.echo("Available Models (OpenAI format):")
        click.echo("=" * 50)
        for model in models.data:
            click.echo(f"  {model.id}")


@models_group.command(name="pull")
@click.argument("model_name")
@click.pass_context
@with_client
def pull_model(ctx: click.Context, client: MLXR, model_name: str) -> None:
    """Pull a model from the registry."""
    click.echo(f"Pulling model: {model_name}")
    for chunk in client.ollama.pull(model_name, stream=True):
        status = chunk.status
        if chunk.total and chunk.completed:
            percent = (chunk.completed / chunk.total) * 100
            click.echo(f"  {status}: {percent:.1f}%")
        else:
            click.echo(f"  {status}")

    click.echo("Model pulled successfully!")


@models_group.command(name="show")
@click.argument("model_name")
@click.pass_context
@with_client
def show_model(ctx: click.Context, client: MLXR, model_name: str) -> None:
    """Show model information."""
    info = client.ollama.show(model_name)
    click.echo(f"Model: {model_name}")
    click.echo("=" * 50)
    click.echo(f"Format: {info.details.format}")
    click.echo(f"Family: {info.details.family}")
    click.echo(f"Parameter size: {info.details.parameter_size}")
    click.echo(f"Quantization: {info.details.quantization_level}")
    click.echo(f"\nTemplate:\n{info.template}")


@models_group.command(name="ps")
@click.pass_context
@with_client
def list_running(ctx: click.Context, client: MLXR) -> None:
    """List running models."""
    running = client.ollama.ps()
    if running.models:
        click.echo("Running Models:")
        click.echo("=" * 50)
        for model in running.models:
            vram_mb = model.size_vram / (1024 * 1024)
            click.echo(f"  {model.name}")
            click.echo(f"    VRAM: {vram_mb:.1f} MB")
            click.echo(f"    Expires: {model.expires_at}")
            click.echo()
    else:
        click.echo("No models currently running")


@main.command()
@click.argument("prompt")
@click.option("-m", "--model", default="TinyLlama-1.1B", help="Model to use")
@click.option("--stream/--no-stream", default=True, help="Stream responses")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--max-tokens", type=int, help="Maximum tokens to generate")
@click.pass_context
@with_client
def chat(
    ctx: click.Context,
    client: MLXR,
    prompt: str,
    model: str,
    stream: bool,
    temperature: float,
    max_tokens: Optional[int],
) -> None:
    """Send a chat message."""
    if stream:
        response_stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                click.echo(chunk.choices[0].delta.content, nl=False)

        click.echo()  # Final newline
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        click.echo(response.choices[0].message.content)

        if response.usage:
            click.echo(
                f"\n(Tokens: {response.usage.total_tokens})",
                err=True,
            )


@main.command()
@click.argument("text")
@click.option("-m", "--model", default="TinyLlama-1.1B", help="Model to use")
@click.pass_context
@with_client
def embed(ctx: click.Context, client: MLXR, text: str, model: str) -> None:
    """Generate embeddings for text."""
    response = client.embeddings.create(model=model, input=text)

    if not response.data:
        click.echo("Error: No embedding data returned", err=True)
        sys.exit(1)

    embedding = response.data[0].embedding

    click.echo(f"Embedding dimension: {len(embedding)}")
    click.echo(f"First 10 values: {embedding[:10]}")

    if response.usage:
        click.echo(f"Tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
