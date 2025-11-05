# MLXR Test Suite

## Overview

This directory contains tests for MLXR at different levels:

- `unit/` - Unit tests for individual components (C++ with Google Test, Python with pytest)
- `integration/` - Integration tests for component interactions
- `e2e/` - End-to-end tests for full system functionality

## Phase 0 Validation

For Phase 0 (Foundation), we have basic validation tests:

### Running Phase 0 Validation

```bash
# From project root
python3 tests/test_metal_compilation.py
```

This validates:
- Metal build script exists and is executable
- Metal source files exist
- Metal shaders compile successfully
- CMake can configure the project

## Future Test Structure

### Unit Tests (Phase 1+)

C++ unit tests using Google Test:
```bash
cd build
cmake .. -DBUILD_TESTS=ON
make
ctest
```

Python unit tests using pytest:
```bash
pytest tests/unit/
```

### Integration Tests (Phase 2+)

```bash
pytest tests/integration/
```

### E2E Tests (Phase 3+)

End-to-end tests using Playwright for GUI:
```bash
cd app/ui
npm run test:e2e
```

## Test Guidelines

1. **Write tests first** - TDD approach for new features
2. **Test at appropriate level** - Unit tests for components, integration for interfaces, E2E for workflows
3. **Fast tests** - Unit tests should run in milliseconds
4. **Isolated tests** - Tests should not depend on each other
5. **Clear assertions** - Test names and assertions should be self-documenting

## Coverage

We aim for:
- Core components: 80%+ coverage
- API endpoints: 90%+ coverage
- Critical paths (KV cache, attention): 95%+ coverage

Coverage reports:
```bash
# C++
cmake .. -DENABLE_COVERAGE=ON
make coverage

# Python
pytest --cov=mlxrunner tests/
```

## Performance Tests

Performance benchmarks live in `tests/benchmarks/` and track:
- Tokens/second (prefill and decode)
- Latency (p50, p95, p99)
- Memory usage
- GPU utilization

Run benchmarks:
```bash
python tests/benchmarks/run_all.py
```

## Continuous Integration

Tests run automatically on:
- Every commit (smoke tests)
- Pull requests (full suite)
- Nightly (extended tests + performance regression)

See `.github/workflows/` for CI configuration.
