"""
Command-line interface for MLXR.

Provides CLI commands for model management, inference, and server operations.
"""

import click


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """MLXR - High-performance LLM inference for Apple Silicon."""
    pass


@main.command()
def status() -> None:
    """Show MLXR status and configuration."""
    click.echo("MLXR Status")
    click.echo("===========")
    click.echo("Version: 0.1.0")
    click.echo("Status: Phase 0 (Foundation) complete, Phase 1 in progress")
    click.echo("")
    click.echo("Run 'make status' for detailed environment information.")


@main.command()
@click.argument("model_name")
def pull(model_name: str) -> None:
    """Download a model from the registry."""
    click.echo(f"Pulling model: {model_name}")
    click.echo("Not implemented yet (Phase 2+)")


@main.command()
def serve() -> None:
    """Start the MLXR inference server."""
    click.echo("Starting MLXR server...")
    click.echo("Not implemented yet (Phase 2+)")


if __name__ == "__main__":
    main()
