# Changelog

All notable changes to the MLXR TypeScript SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-06

### Added
- Initial release of MLXR TypeScript SDK
- OpenAI-compatible API client
  - Chat completions (streaming and non-streaming)
  - Text completions (streaming and non-streaming)
  - Embeddings
  - Model listing and info
- Ollama-compatible API client
  - Chat (streaming and non-streaming)
  - Generate (streaming and non-streaming)
  - Embeddings
  - Model management (pull, create, tags, ps, show, delete, copy)
- Unix domain socket support for macOS
- HTTP/HTTPS support
- SSE streaming support
- Full TypeScript type definitions
- Comprehensive examples
- Zero external dependencies

### Features
- Automatic connection detection (Unix socket vs HTTP)
- API key authentication support
- Configurable request timeouts
- Custom headers support
- Health check endpoint
- Complete error handling

### Documentation
- Full README with API examples
- TypeScript examples for all major features
- Inline code documentation
