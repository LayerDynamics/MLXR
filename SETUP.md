# MLXR Development Setup Guide

This guide will help you set up the MLXR development environment.

## Prerequisites

### Required
- **macOS 14.0 (Sonoma)** or later
- **Apple Silicon** (M2, M3, or M4)
- **Xcode 15+** with Command Line Tools
- **Python 3.11+**
- **CMake 3.20+**
- **Git**

### Recommended
- **Homebrew** for package management
- **Conda** or **venv** for Python environment management

## Quick Start

### 1. Install System Dependencies

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake and other tools
brew install cmake ninja
```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd MLXR
```

### 3. Set Up Python Environment

#### Option A: Using venv (Recommended for simplicity)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# For development (includes testing, linting tools)
pip install -r requirements-dev.txt
```

#### Option B: Using Conda

```bash
# Create environment from file
conda env create -f environment.yml

# Activate it
conda activate mlxr
```

### 4. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up git hooks
pre-commit install
```

### 5. Verify Installation

```bash
# Run Phase 0 validation tests
python3 tests/test_metal_compilation.py

# Or use Make
make test-phase0
```

You should see:
```
✓ Metal build script exists and is executable
✓ Metal shaders compiled successfully
✓ CMake configuration successful
=== All validation tests passed! ===
```

## Development Workflow

### Using Make (Recommended)

```bash
# See all available commands
make help

# Initial setup (creates venv and installs deps)
make setup

# Build Metal shaders
make metal

# Configure and build C++ components
make build

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

### Manual Commands

```bash
# Compile Metal shaders
./scripts/build_metal.sh

# Configure CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Run tests
pytest tests/

# Format Python code
black .
ruff check --fix .
```

## Verifying MLX Installation

```bash
python3 -c "import mlx; print(f'MLX version: {mlx.__version__}')"
```

If MLX is not installed:
```bash
pip install mlx mlx-lm
```

**Note**: MLX only works on Apple Silicon. If you're on Intel Mac, you'll get an error.

## IDE Setup

### VSCode (Recommended)

Install these extensions:
- Python (Microsoft)
- C/C++ (Microsoft)
- CMake Tools
- Metal Shader Language Support

Suggested settings ([.vscode/settings.json]()):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
  "files.associations": {
    "*.metal": "metal"
  }
}
```

### CLion / Xcode

Both work well with the CMake project structure. Open the root `CMakeLists.txt`.

## Troubleshooting

### "MLX not found" during CMake configuration

MLX paths are set in the root `CMakeLists.txt`. If your MLX installation is in a different location:

```bash
cmake .. -DMLX_INCLUDE_DIR=/path/to/mlx/include -DMLX_LIBRARY_DIR=/path/to/mlx/lib
```

### Metal compilation fails

Ensure Xcode Command Line Tools are installed:
```bash
xcode-select --install
```

Check Metal compiler:
```bash
xcrun -sdk macosx metal --version
```

### Python version mismatch

MLXR requires Python 3.11+. Check your version:
```bash
python3 --version
```

If needed, install Python 3.11 via Homebrew:
```bash
brew install python@3.11
```

### Permission denied on scripts

Make scripts executable:
```bash
chmod +x scripts/*.sh
chmod +x tests/*.py
```

## Next Steps

Once setup is complete:

1. **Phase 0**: ✅ Complete (foundation)
2. **Phase 1**: Start implementing MLX integration
   - See [CLAUDE.md](CLAUDE.md) for architecture details
   - Check the current branch: `git branch`
   - Follow the Phase 1 checklist in the todo list

## Getting Help

- Check [CLAUDE.md](CLAUDE.md) for detailed architecture
- Review [plan/SPEC01.md](plan/SPEC01.md) for complete specification
- See [tests/README.md](tests/README.md) for testing guidelines

## Common Tasks

```bash
# Start new feature branch
git checkout -b feature-name

# Build and test
make build && make test

# Format before committing
make format

# Clean and rebuild
make clean && make build
```

---

**Ready to code!** 🚀 Proceed to Phase 1 implementation.
