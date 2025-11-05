# MLXR Development Makefile

.PHONY: help setup install install-dev clean build test format lint metal cmake

# Default target
help:
	@echo "MLXR Development Commands:"
	@echo ""
	@echo "  setup         - Initial setup (create venv, install dependencies)"
	@echo "  install       - Install Python package in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  metal         - Compile Metal shaders"
	@echo "  cmake         - Configure CMake build"
	@echo "  build         - Full build (Metal + CMake)"
	@echo "  test          - Run all tests"
	@echo "  test-phase0   - Run Phase 0 validation tests"
	@echo "  format        - Format code (black, clang-format)"
	@echo "  lint          - Lint code (ruff, mypy)"
	@echo "  clean         - Clean build artifacts"
	@echo "  clean-all     - Clean everything including venv"
	@echo ""

# Setup Python environment
setup:
	@echo "Setting up Python environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete! Activate with: source venv/bin/activate"

# Install package
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,server]"
	pre-commit install

# Build Metal shaders
metal:
	@echo "Compiling Metal shaders..."
	./scripts/build_metal.sh

# Configure CMake
cmake:
	@echo "Configuring CMake..."
	mkdir -p build/cmake
	cd build/cmake && cmake ../.. -DCMAKE_BUILD_TYPE=Release

# Full build
build: metal cmake
	@echo "Building C++ components..."
	cd build/cmake && make -j8

# Testing
test:
	pytest tests/ -v

test-phase0:
	python3 tests/test_metal_compilation.py

# Code formatting
format:
	@echo "Formatting Python code..."
	black .
	ruff check --fix .
	@echo "Formatting C++ code..."
	find core daemon -name "*.cpp" -o -name "*.h" -o -name "*.cc" -o -name "*.hpp" | xargs clang-format -i || true

# Linting
lint:
	@echo "Linting Python..."
	ruff check .
	mypy sdks/python --ignore-missing-imports || true
	@echo "Linting C++..."
	# TODO: Add clang-tidy when code is implemented

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true
	find . -type f -name "*.metallib" -delete || true
	find . -type f -name "*.air" -delete || true

# Clean everything including venv
clean-all: clean
	rm -rf venv/
	@echo "✓ All clean!"

# Development shortcuts
dev: install-dev metal
	@echo "✓ Development environment ready!"

# Quick validation
validate: test-phase0
	@echo "✓ Phase 0 validation passed!"
