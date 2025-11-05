# MLXR Development Makefile

.PHONY: help setup install install-dev clean build test format lint metal cmake

# Default target
help:
	@echo "MLXR Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  status        - Check environment and setup status"
	@echo "  setup         - Initial setup with Conda (recommended)"
	@echo "  setup-venv    - Alternative setup with virtualenv"
	@echo ""
	@echo "Build:"
	@echo "  metal         - Compile Metal shaders"
	@echo "  cmake         - Configure CMake build"
	@echo "  build         - Full build (Metal + CMake)"
	@echo "  dev           - Quick dev setup (Metal only, for Phase 0-1)"
	@echo "  dev-full      - Full dev setup (includes package install)"
	@echo ""
	@echo "Install:"
	@echo "  install       - Install Python package in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run all tests"
	@echo "  test-phase0   - Run Phase 0 validation tests"
	@echo "  validate      - Quick validation"
	@echo ""
	@echo "Code Quality:"
	@echo "  format        - Format code (black, clang-format)"
	@echo "  lint          - Lint code (ruff, mypy)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         - Clean build artifacts"
	@echo "  clean-all     - Clean everything including conda env"
	@echo ""

# Check environment and setup status
status:
	@echo "=== MLXR Environment Status ==="
	@echo ""
	@echo "System:"
	@uname -m | awk '{print "  Architecture: " $$0}'
	@sw_vers -productVersion | awk '{print "  macOS Version: " $$0}'
	@echo ""
	@echo "Build Tools:"
	@if command -v xcrun &> /dev/null; then \
		echo "  ✓ Xcode Command Line Tools installed"; \
		xcrun metal --version 2>&1 | head -1 | sed 's/^/    /'; \
	else \
		echo "  ❌ Xcode Command Line Tools NOT found"; \
	fi
	@if command -v cmake &> /dev/null; then \
		cmake --version | head -1 | sed 's/^/  ✓ /'; \
	else \
		echo "  ❌ CMake NOT found"; \
	fi
	@echo ""
	@echo "Python Environment:"
	@if command -v conda &> /dev/null; then \
		echo "  ✓ Conda installed"; \
		if conda env list | grep -q "^mlxr "; then \
			echo "  ✓ Conda environment 'mlxr' exists"; \
			if [ "$$CONDA_DEFAULT_ENV" = "mlxr" ]; then \
				echo "  ✓ Environment 'mlxr' is ACTIVATED"; \
			else \
				echo "  ⚠️  Environment 'mlxr' exists but NOT activated"; \
				echo "     Run: conda activate mlxr"; \
			fi; \
		else \
			echo "  ⚠️  Conda environment 'mlxr' NOT found"; \
			echo "     Run: make setup"; \
		fi; \
	else \
		echo "  ⚠️  Conda not installed"; \
	fi
	@if [ -d "venv" ]; then \
		echo "  ✓ Virtualenv 'venv' exists"; \
		if [ -n "$$VIRTUAL_ENV" ]; then \
			echo "  ✓ Virtualenv is ACTIVATED"; \
		else \
			echo "  ⚠️  Virtualenv exists but NOT activated"; \
			echo "     Run: source venv/bin/activate"; \
		fi; \
	fi
	@if [ -n "$$CONDA_DEFAULT_ENV" ] || [ -n "$$VIRTUAL_ENV" ]; then \
		echo ""; \
		echo "Active Environment:"; \
		python3 --version | sed 's/^/  /'; \
		if python3 -c "import mlx" 2>/dev/null; then \
			python3 -c "import mlx; print('  ✓ MLX installed: ' + mlx.__version__)"; \
		else \
			echo "  ❌ MLX NOT installed"; \
		fi; \
	fi
	@echo ""
	@echo "Build Status:"
	@if [ -d "build/lib" ] && [ -n "$$(ls -A build/lib/*.metallib 2>/dev/null)" ]; then \
		echo "  ✓ Metal kernels compiled"; \
		ls build/lib/*.metallib | wc -l | awk '{print "    Found " $$1 " metallib file(s)"}'; \
	else \
		echo "  ⚠️  Metal kernels NOT compiled"; \
		echo "     Run: make metal"; \
	fi
	@echo ""

# Setup Python environment (conda - recommended for ML)
setup:
	@echo "Setting up Python environment with Conda..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda not found. Please install Miniconda or Anaconda first."; \
		echo "   Or use 'make setup-venv' for virtualenv setup."; \
		exit 1; \
	fi
	@if conda env list | grep -q "^mlxr "; then \
		echo "⚠️  Conda environment 'mlxr' already exists. Updating..."; \
		conda env update -f environment.yml --prune; \
	else \
		conda env create -f environment.yml; \
	fi
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. conda activate mlxr"
	@echo "  2. make dev           # Install dev tools and compile Metal kernels"
	@echo "  3. make test-phase0   # Validate the build system"

# Alternative: Setup with virtualenv (if conda not available)
setup-venv:
	@echo "Setting up Python environment with venv..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete! Activate with: source venv/bin/activate"

# Install package (checks if environment is activated)
install:
	@if [ -z "$$CONDA_DEFAULT_ENV" ] && [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ No Python environment activated!"; \
		echo "   Activate conda: conda activate mlxr"; \
		echo "   Or activate venv: source venv/bin/activate"; \
		exit 1; \
	fi
	pip install -e .

install-dev:
	@if [ -z "$$CONDA_DEFAULT_ENV" ] && [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ No Python environment activated!"; \
		echo "   Activate conda: conda activate mlxr"; \
		echo "   Or activate venv: source venv/bin/activate"; \
		exit 1; \
	fi
	pip install -e ".[dev,server]" && python3 -m pre_commit install

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

# Clean everything including conda env and venv
clean-all: clean
	@echo "Cleaning environments..."
	@if command -v conda &> /dev/null && conda env list | grep -q "^mlxr "; then \
		echo "Removing conda environment 'mlxr'..."; \
		conda env remove -n mlxr -y; \
	fi
	@if [ -d "venv" ]; then \
		echo "Removing venv..."; \
		rm -rf venv/; \
	fi
	@echo "✓ All clean!"

# Development shortcuts (for early phases - skip package install)
dev: metal
	@echo "✓ Development environment ready!"
	@echo ""
	@echo "Note: Package install skipped (not needed for Phase 0-1)"
	@echo "      Use 'make install-dev' once Phase 1 implementation begins"

# Full development setup (includes package installation)
dev-full: install-dev metal
	@echo "✓ Full development environment ready!"

# Quick validation
validate: test-phase0
	@echo "✓ Phase 0 validation passed!"
