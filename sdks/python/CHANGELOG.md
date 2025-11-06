# Changelog

All notable changes to the MLXR Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-06

### Added

#### Core Features
- **Main Client (`MLXR`)**: Synchronous client with dual API support
- **Async Client (`AsyncMLXR`)**: Full async/await support with AsyncIterator streaming
- **Transport Layer**: Unix Domain Socket and HTTP connection support with automatic fallback

#### OpenAI-Compatible API
- Chat completions endpoint (`/v1/chat/completions`)
  - Non-streaming and streaming (SSE) support
  - Full parameter support (temperature, top_p, max_tokens, etc.)
- Text completions endpoint (`/v1/completions`)
- Embeddings endpoint (`/v1/embeddings`)
- Models listing and retrieval (`/v1/models`)

#### Ollama-Compatible API
- Generate endpoint (`/api/generate`)
- Chat endpoint (`/api/chat`)
- Embeddings endpoint (`/api/embeddings`)
- Model management:
  - Pull models (`/api/pull`)
  - Create models (`/api/create`)
  - Copy models (`/api/copy`)
  - Delete models (`/api/delete`)
  - Show model info (`/api/show`)
  - List models (`/api/tags`)
  - List running models (`/api/ps`)

#### Type System
- Comprehensive Pydantic models for all API types
- Full type hints throughout the codebase
- PEP 561 compatibility with `py.typed` marker

#### Error Handling
- Custom exception hierarchy
- Proper HTTP status code mapping
- Connection and timeout error handling
- Validation error support

#### CLI (`mlxr` command)
- `mlxr status` - Daemon health and metrics
- `mlxr models list` - List available models
- `mlxr models pull <name>` - Pull models from registry
- `mlxr models show <name>` - Show model information
- `mlxr models ps` - List running models
- `mlxr chat <prompt>` - Interactive chat with streaming
- `mlxr embed <text>` - Generate embeddings
- Global options: `--base-url`, `--api-key`

#### Examples
- `basic_chat.py` - Simple chat completion
- `streaming_chat.py` - SSE streaming demonstration
- `ollama_generate.py` - Ollama API usage
- `model_management.py` - Model operations
- `async_chat.py` - Async/await patterns
- `embeddings.py` - Embedding generation
- `health_and_metrics.py` - Monitoring and observability

#### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Examples README with usage patterns
- Type documentation via docstrings

#### Development Tools
- `pyproject.toml` with modern Python packaging
- Black, Ruff, and MyPy configuration
- pytest setup with async support
- Development dependencies specification

### Implementation Details

#### Transport
- Custom `UDSTransport` for Unix Domain Socket connections
- `httpx`-based HTTP transport with connection pooling
- SSE (Server-Sent Events) streaming support
- Automatic JSON parsing and error handling
- Request/response timeout configuration

#### Streaming
- Iterator-based streaming for synchronous client
- AsyncIterator-based streaming for async client
- Proper SSE format parsing (`data: {...}` and `[DONE]`)
- Support for both OpenAI and Ollama streaming formats

#### API Compatibility
- Full OpenAI Chat Completion API parity
- Full Ollama API parity
- Proper request/response type validation
- Usage statistics tracking

### Dependencies
- `httpx>=0.24.0` - Modern HTTP client with HTTP/2 and async support
- `pydantic>=2.0.0` - Data validation and settings management
- `click>=8.0.0` - CLI framework
- `typing-extensions>=4.5.0` - Backported type hints

### Known Limitations
- Requires MLXR daemon to be running
- macOS with Apple Silicon only (M1/M2/M3/M4)
- Python 3.11+ required
- No Windows or Linux support (daemon limitation)

### Notes
- This is an alpha release (0.1.0)
- API may change in future versions
- Feedback and contributions welcome

## [Unreleased]

### Planned Features
- Batch request support
- Function calling support
- Vision model support (image inputs)
- Model conversion utilities
- Streaming cancellation
- Request retries with exponential backoff
- Connection pooling optimizations
- Comprehensive test suite
- API usage examples for all endpoints
- Performance benchmarks

[0.1.0]: https://github.com/LayerDynamics/MLXR/releases/tag/v0.1.0
[Unreleased]: https://github.com/LayerDynamics/MLXR/compare/v0.1.0...HEAD
