#!/bin/bash
# Install Homebrew dependencies for MLXR
# This script is used by both the Makefile and CI/CD workflows

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Installing MLXR Homebrew Dependencies ==="

# Core dependencies (always required)
CORE_DEPS=(
    "mlx"
    "sentencepiece"
    "nlohmann-json"
    "cpp-httplib"
    "googletest"
)

# Build tool dependencies (optional, can be skipped in CI with pre-installed tools)
BUILD_DEPS=(
    "cmake"
    "ninja"
)

# Additional dependencies based on environment
EXTRA_DEPS=()

# Parse command line arguments
INSTALL_BUILD_TOOLS=false
INSTALL_EXTRAS=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-tools)
            INSTALL_BUILD_TOOLS=true
            shift
            ;;
        --with-node)
            EXTRA_DEPS+=("node")
            shift
            ;;
        --with-apache-bench)
            EXTRA_DEPS+=("apache-bench")
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--build-tools] [--with-node] [--with-apache-bench] [--quiet]"
            exit 1
            ;;
    esac
done

# Function to check if a package is installed
is_installed() {
    brew list "$1" &>/dev/null
}

# Function to install a package if not already installed
install_if_needed() {
    local package=$1
    if is_installed "$package"; then
        if [ "$QUIET" = false ]; then
            echo -e "${GREEN}✓${NC} $package already installed"
        fi
    else
        echo -e "${YELLOW}→${NC} Installing $package..."
        brew install "$package"
    fi
}

# Install core dependencies
echo ""
echo "Installing core dependencies..."
for dep in "${CORE_DEPS[@]}"; do
    install_if_needed "$dep"
done

# Install build tools if requested
if [ "$INSTALL_BUILD_TOOLS" = true ]; then
    echo ""
    echo "Installing build tools..."
    for dep in "${BUILD_DEPS[@]}"; do
        install_if_needed "$dep"
    done
fi

# Install extra dependencies
if [ ${#EXTRA_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "Installing additional dependencies..."
    for dep in "${EXTRA_DEPS[@]}"; do
        install_if_needed "$dep"
    done
fi

echo ""
echo -e "${GREEN}✓${NC} All dependencies installed successfully"
