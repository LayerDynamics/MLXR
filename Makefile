# MLXR Development Makefile

.PHONY: help status install-deps setup install install-dev clean build test format lint metal cmake test-cpp test-cpp-verbose test-all app-xcode app-ui app app-dev app-run app-sign app-dmg app-release app-setup-test app-test app-test-verbose app-test-coverage app-test-only app-test-suite app-test-open app-test-check app-test-clean app-test-all

# Default target
help:
	@echo "MLXR Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  status        - Check environment and setup status"
	@echo "  install-deps  - Install system dependencies via Homebrew"
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
	@echo "  test          - Run Python tests (pytest)"
	@echo "  test-cpp      - Run C++ unit tests"
	@echo "  test-cpp-verbose - Run C++ unit tests with verbose output"
	@echo "  test-all      - Run all tests (C++ and Python)"
	@echo "  test-phase0   - Run Phase 0 validation tests"
	@echo "  validate      - Quick validation"
	@echo ""
	@echo "Code Quality:"
	@echo "  format        - Format code (black, clang-format)"
	@echo "  lint          - Lint code (ruff, mypy)"
	@echo ""
	@echo "macOS App - Build:"
	@echo "  app-xcode     - Setup Xcode project (using XcodeGen)"
	@echo "  app-ui        - Build React UI"
	@echo "  app           - Build complete macOS app"
	@echo "  app-dev       - Build app in development mode"
	@echo "  app-run       - Build and run app"
	@echo "  app-sign      - Sign the app"
	@echo "  app-dmg       - Create DMG installer"
	@echo "  app-release   - Full release build (build + sign + dmg)"
	@echo ""
	@echo "macOS App - Testing:"
	@echo "  app-setup-test    - Complete test environment setup"
	@echo "  app-test          - Run all automated tests"
	@echo "  app-test-verbose  - Run tests with verbose output"
	@echo "  app-test-coverage - Run tests with code coverage"
	@echo "  app-test-only     - Run specific test suite (SUITE=BridgeTests)"
	@echo "  app-test-suite    - List available test suites"
	@echo "  app-test-open     - Open Xcode for interactive testing"
	@echo "  app-test-check    - Validate test environment"
	@echo "  app-test-clean    - Clean test artifacts"
	@echo "  app-test-all      - Full test workflow (setup + test)"
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
	@echo "System Dependencies (Homebrew):"
	@if command -v brew &> /dev/null; then \
		echo "  ✓ Homebrew installed"; \
		if brew list sentencepiece &> /dev/null; then \
			echo "  ✓ sentencepiece"; \
		else \
			echo "  ❌ sentencepiece (run: brew install sentencepiece)"; \
		fi; \
		if brew list nlohmann-json &> /dev/null; then \
			echo "  ✓ nlohmann-json"; \
		else \
			echo "  ❌ nlohmann-json (run: brew install nlohmann-json)"; \
		fi; \
		if brew list cpp-httplib &> /dev/null; then \
			echo "  ✓ cpp-httplib"; \
		else \
			echo "  ❌ cpp-httplib (run: brew install cpp-httplib)"; \
		fi; \
		if brew list googletest &> /dev/null; then \
			echo "  ✓ googletest"; \
		else \
			echo "  ❌ googletest (run: brew install googletest)"; \
		fi; \
	else \
		echo "  ❌ Homebrew not installed"; \
		echo "     Install from: https://brew.sh"; \
	fi
	@echo ""

# Install system dependencies via Homebrew
install-deps:
	@echo "Installing system dependencies via Homebrew..."
	@if ! command -v brew &> /dev/null; then \
		echo "❌ Homebrew not found. Install from: https://brew.sh"; \
		exit 1; \
	fi
	@echo "Installing dependencies using centralized script..."
	@./scripts/install_homebrew_deps.sh --build-tools
	@echo ""
	@echo "✓ System dependencies installed!"
	@echo ""
	@echo "Next: make status  # Verify installation"

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

test-cpp:
	@if [ ! -f "build/cmake/bin/mlxr_unit_tests" ]; then \
		echo "❌ C++ unit tests not built. Run 'make build' first."; \
		exit 1; \
	fi
	@echo "Running C++ unit tests..."
	cd build/cmake && ctest --output-on-failure

test-cpp-verbose:
	@if [ ! -f "build/cmake/bin/mlxr_unit_tests" ]; then \
		echo "❌ C++ unit tests not built. Run 'make build' first."; \
		exit 1; \
	fi
	@echo "Running C++ unit tests (verbose)..."
	./build/cmake/bin/mlxr_unit_tests --gtest_color=yes

test-all: test-cpp test
	@echo ""
	@echo "✓ All tests passed!"

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
validate: test-phase0 test-cpp
	@echo ""
	@echo "✓ Phase 1 validation passed!"

# ============================================================================
# macOS App Build Targets
# ============================================================================

# Setup Xcode project (using XcodeGen)
app-xcode:
	@echo "Setting up Xcode project..."
	./scripts/setup_xcode_project.sh

# Build frontend (React UI)
app-ui:
	@echo "Building React UI..."
	cd app/ui && npm install && npm run build
	@echo "✓ React UI built"

# Build macOS app
app: app-ui
	@echo "Building MLXR.app..."
	./scripts/build_app.sh
	@echo "✓ MLXR.app built"

# Build and run app
app-run: app
	@echo "Launching MLXR.app..."
	open build/macos/MLXR.app

# Build app in development mode (loads from dev server)
app-dev:
	@echo "Building MLXR.app (development mode)..."
	MLXR_DEV_MODE=1 ./scripts/build_app.sh Debug
	@echo ""
	@echo "Start the dev server in another terminal:"
	@echo "  cd app/ui && npm run dev"
	@echo ""
	@echo "Then run the app:"
	@echo "  make app-run"

# Sign the app
app-sign: app
	@echo "Signing MLXR.app..."
	./scripts/sign_app.sh
	@echo "✓ App signed"

# Create DMG
app-dmg: app
	@echo "Creating DMG..."
	./scripts/create_dmg.sh
	@echo "✓ DMG created"

# Full app release build (build + sign + dmg)
app-release: app-sign app-dmg
	@echo "✓ Release build complete!"
	@echo ""
	@echo "Artifacts:"
	@ls -lh build/*.dmg

# ============================================================================
# macOS App Testing Targets
# ============================================================================

# Complete setup for testing (Xcode + UI + Daemon)
app-setup-test: app-xcode app-ui build
	@echo ""
	@echo "✓ Test environment setup complete!"
	@echo ""
	@echo "Ready to run tests:"
	@echo "  make app-test          - Run all automated tests"
	@echo "  make app-test-verbose  - Run tests with verbose output"
	@echo "  make app-test-coverage - Run tests with code coverage report"

# Run automated tests (requires app-xcode first)
app-test:
	@echo "Running MLXR app tests..."
	@if [ ! -d "app/macos/MLXR.xcodeproj" ]; then \
		echo "❌ Xcode project not found. Run 'make app-xcode' first."; \
		exit 1; \
	fi
	@xcodebuild test \
		-project app/macos/MLXR.xcodeproj \
		-scheme MLXR \
		-destination 'platform=macOS,arch=arm64' \
		-quiet
	@echo "✓ All tests passed!"

# Run tests with verbose output
app-test-verbose:
	@echo "Running MLXR app tests (verbose)..."
	@if [ ! -d "app/macos/MLXR.xcodeproj" ]; then \
		echo "❌ Xcode project not found. Run 'make app-xcode' first."; \
		exit 1; \
	fi
	xcodebuild test \
		-project app/macos/MLXR.xcodeproj \
		-scheme MLXR \
		-destination 'platform=macOS,arch=arm64'

# Run tests with code coverage
app-test-coverage:
	@echo "Running MLXR app tests with code coverage..."
	@if [ ! -d "app/macos/MLXR.xcodeproj" ]; then \
		echo "❌ Xcode project not found. Run 'make app-xcode' first."; \
		exit 1; \
	fi
	xcodebuild test \
		-project app/macos/MLXR.xcodeproj \
		-scheme MLXR \
		-destination 'platform=macOS,arch=arm64' \
		-enableCodeCoverage YES
	@echo ""
	@echo "✓ Tests completed with coverage data!"
	@echo ""
	@echo "To view coverage report:"
	@echo "  1. Open app/macos/MLXR.xcodeproj in Xcode"
	@echo "  2. Go to Report Navigator (Cmd+9)"
	@echo "  3. Select latest test run"
	@echo "  4. Click 'Coverage' tab"

# Run specific test suite
app-test-suite:
	@echo "Available test suites:"
	@echo "  1. BridgeTests"
	@echo "  2. DaemonManagerTests"
	@echo "  3. ServicesTests"
	@echo "  4. IntegrationTests"
	@echo ""
	@echo "Usage: make app-test-only SUITE=BridgeTests"

# Run specific test (requires SUITE variable)
app-test-only:
	@if [ -z "$(SUITE)" ]; then \
		echo "❌ Error: SUITE variable required"; \
		echo "Usage: make app-test-only SUITE=BridgeTests"; \
		exit 1; \
	fi
	@echo "Running $(SUITE)..."
	xcodebuild test \
		-project app/macos/MLXR.xcodeproj \
		-scheme MLXR \
		-destination 'platform=macOS,arch=arm64' \
		-only-testing:MLXRTests/$(SUITE)

# Open test results in Xcode
app-test-open:
	@echo "Opening Xcode for testing..."
	@if [ ! -d "app/macos/MLXR.xcodeproj" ]; then \
		echo "❌ Xcode project not found. Run 'make app-xcode' first."; \
		exit 1; \
	fi
	open app/macos/MLXR.xcodeproj
	@echo ""
	@echo "To run tests in Xcode:"
	@echo "  1. Press Cmd+U to run all tests"
	@echo "  2. Or use Test Navigator (Cmd+6) to run specific tests"

# Validate test environment
app-test-check:
	@echo "=== Test Environment Check ==="
	@echo ""
	@echo "Xcode Project:"
	@if [ -d "app/macos/MLXR.xcodeproj" ]; then \
		echo "  ✓ Xcode project exists"; \
	else \
		echo "  ❌ Xcode project missing (run: make app-xcode)"; \
	fi
	@echo ""
	@echo "React UI:"
	@if [ -d "app/ui/dist" ]; then \
		echo "  ✓ React UI built"; \
	else \
		echo "  ❌ React UI not built (run: make app-ui)"; \
	fi
	@echo ""
	@echo "Daemon Binary:"
	@if [ -f "build/cmake/bin/mlxrunnerd" ]; then \
		echo "  ✓ Daemon binary exists"; \
	else \
		echo "  ⚠️  Daemon binary missing (run: make build)"; \
		echo "     Note: Some tests may fail without daemon"; \
	fi
	@echo ""
	@echo "Test Files:"
	@TEST_COUNT=$$(find app/macos/MLXRTests -name "*.swift" 2>/dev/null | wc -l | tr -d ' '); \
	if [ $$TEST_COUNT -gt 0 ]; then \
		echo "  ✓ $$TEST_COUNT test files found"; \
	else \
		echo "  ❌ No test files found"; \
	fi

# Clean test artifacts
app-test-clean:
	@echo "Cleaning test artifacts..."
	@rm -rf app/macos/build
	@rm -rf app/macos/DerivedData
	@echo "✓ Test artifacts cleaned"

# Full test workflow (setup + run tests)
app-test-all: app-setup-test app-test
	@echo ""
	@echo "✓ Complete test workflow finished!"
	@ls -lh build/macos/MLXR.app

# Clean app build artifacts
app-clean:
	@echo "Cleaning app build artifacts..."
	rm -rf build/macos
	rm -f build/*.dmg
	cd app/ui && rm -rf dist node_modules/.vite
	@echo "✓ App artifacts cleaned"

.PHONY: app-ui app app-run app-dev app-sign app-dmg app-release app-clean
