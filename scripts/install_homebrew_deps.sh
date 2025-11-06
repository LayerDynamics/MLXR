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

# Check if Homebrew is installed
if ! command -v brew &>/dev/null; then
    echo -e "${RED}✗${NC} Homebrew is not installed!"
    echo ""
    echo "Please install Homebrew first:"
    echo "  https://brew.sh"
    echo ""
    echo "Or run:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Homebrew detected: $(brew --version | head -1)"

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

# Function to install a package if not already installed (with retry logic)
install_if_needed() {
    local package=$1
    if is_installed "$package"; then
        if [ "$QUIET" = false ]; then
            echo -e "${GREEN}✓${NC} $package already installed"
        fi
    else
        echo -e "${YELLOW}→${NC} Installing $package..."
        local max_retries=3
        local attempt=1
        local success=false

        while [ $attempt -le $max_retries ]; do
            if brew install "$package" 2>&1; then
                success=true
                echo -e "${GREEN}✓${NC} Successfully installed $package"
                break
            else
                echo -e "${RED}✗${NC} Failed to install $package (attempt $attempt/$max_retries)"
                if [ $attempt -lt $max_retries ]; then
                    echo "Retrying in 2 seconds..."
                    sleep 2
                fi
            fi
            attempt=$((attempt + 1))
        done

        if [ "$success" = false ]; then
            echo -e "${RED}✗${NC} Could not install $package after $max_retries attempts."
            echo "Please check your network connection and try again."
            exit 1
        fi
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
