#!/bin/bash
# Metal shader compilation script for MLXR
# Compiles .metal source files into .metallib libraries

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
METAL_SRC_DIR="$PROJECT_ROOT/core/kernels/metal"
BUILD_DIR="$PROJECT_ROOT/build/metal"
OUTPUT_DIR="$PROJECT_ROOT/build/lib"

# Metal compiler settings
METAL_SDK="macosx"
METAL_STD="metal3.0"
MIN_OS_VERSION="14.0"

echo -e "${GREEN}=== MLXR Metal Shader Build ===${NC}"
echo "Source dir: $METAL_SRC_DIR"
echo "Build dir:  $BUILD_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if xcrun is available
if ! command -v xcrun &> /dev/null; then
    echo -e "${RED}Error: xcrun not found. Please install Xcode Command Line Tools.${NC}"
    exit 1
fi

# Function to compile a single Metal file
compile_metal() {
    local source_file="$1"
    local filename=$(basename "$source_file" .metal)
    local air_file="$BUILD_DIR/${filename}.air"
    local metallib_file="$OUTPUT_DIR/${filename}.metallib"

    echo -e "${YELLOW}Compiling: ${filename}.metal${NC}"

    # Compile .metal to .air (intermediate representation)
    xcrun -sdk "$METAL_SDK" metal \
        -std="$METAL_STD" \
        -mmacosx-version-min="$MIN_OS_VERSION" \
        -c "$source_file" \
        -o "$air_file"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to compile ${filename}.metal${NC}"
        return 1
    fi

    # Link .air to .metallib
    xcrun -sdk "$METAL_SDK" metallib \
        "$air_file" \
        -o "$metallib_file"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to link ${filename}.metallib${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Built: ${filename}.metallib${NC}"
    return 0
}

# Function to compile all Metal files in a directory
compile_all() {
    local compiled_count=0
    local failed_count=0

    # Find all .metal files
    if [ ! -d "$METAL_SRC_DIR" ]; then
        echo -e "${YELLOW}Warning: Metal source directory not found: $METAL_SRC_DIR${NC}"
        echo -e "${YELLOW}Creating directory...${NC}"
        mkdir -p "$METAL_SRC_DIR"
        return 0
    fi

    local metal_files=$(find "$METAL_SRC_DIR" -name "*.metal")

    if [ -z "$metal_files" ]; then
        echo -e "${YELLOW}No .metal files found in $METAL_SRC_DIR${NC}"
        return 0
    fi

    # Compile each file
    for metal_file in $metal_files; do
        if compile_metal "$metal_file"; then
            ((compiled_count++))
        else
            ((failed_count++))
        fi
    done

    echo ""
    echo -e "${GREEN}=== Build Summary ===${NC}"
    echo "Compiled: $compiled_count"
    echo "Failed:   $failed_count"

    if [ $failed_count -gt 0 ]; then
        echo -e "${RED}Build completed with errors${NC}"
        return 1
    else
        echo -e "${GREEN}Build completed successfully${NC}"
        return 0
    fi
}

# Main execution
compile_all

exit $?
