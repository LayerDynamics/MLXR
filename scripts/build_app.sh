#!/bin/bash
#
# build_app.sh
# Build MLXR.app for macOS
#
# This script:
# 1. Builds the React UI (Vite)
# 2. Generates Xcode project (if needed)
# 3. Builds the macOS app
# 4. Bundles all resources (UI, daemon, configs)
# 5. Creates a distributable app bundle
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Building MLXR.app               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$PROJECT_ROOT/app/macos"
UI_DIR="$PROJECT_ROOT/app/ui"
BUILD_DIR="$PROJECT_ROOT/build/macos"
DAEMON_BUILD="$PROJECT_ROOT/build/cmake/bin"
SCHEME="MLXR"
CONFIGURATION="${1:-Release}"  # Default to Release, can override with arg

# ============================================================================
# Step 1: Build React UI
# ============================================================================
echo -e "${YELLOW}[1/6] Building React UI...${NC}"
cd "$UI_DIR"

if [ ! -d "node_modules" ]; then
    echo "  → Installing npm dependencies..."
    npm install --silent
fi

if [ ! -d "dist" ] || [ "$UI_DIR/src" -nt "$UI_DIR/dist" ]; then
    echo "  → Building with Vite..."
    npm run build
else
    echo "  → UI already built (up to date)"
fi

echo -e "${GREEN}  ✓ React UI ready${NC}"
echo ""

# ============================================================================
# Step 2: Generate Xcode project (if needed)
# ============================================================================
echo -e "${YELLOW}[2/6] Checking Xcode project...${NC}"
cd "$APP_DIR"

if [ ! -d "MLXR.xcodeproj" ]; then
    echo "  → Generating Xcode project with xcodegen..."

    if ! command -v xcodegen &> /dev/null; then
        echo -e "${RED}  ✗ xcodegen not found${NC}"
        echo "  Install with: brew install xcodegen"
        exit 1
    fi

    xcodegen generate
    echo -e "${GREEN}  ✓ Xcode project generated${NC}"
else
    echo "  → Xcode project exists"
fi

echo -e "${GREEN}  ✓ Xcode project ready${NC}"
echo ""

# ============================================================================
# Step 3: Build macOS app with xcodebuild
# ============================================================================
echo -e "${YELLOW}[3/6] Building macOS app ($CONFIGURATION)...${NC}"
cd "$APP_DIR"

# Clean derived data for fresh build
if [ -d "$BUILD_DIR" ]; then
    echo "  → Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

echo "  → Running xcodebuild..."
xcodebuild \
    -project MLXR.xcodeproj \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$BUILD_DIR" \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    build 2>&1 | grep -E "(error|warning|Building|Compiling|Linking)" || true

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}  ✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ macOS app built${NC}"
echo ""

# ============================================================================
# Step 4: Bundle resources into app
# ============================================================================
echo -e "${YELLOW}[4/6] Bundling resources...${NC}"

APP_PATH="$BUILD_DIR/Build/Products/$CONFIGURATION/MLXR.app"
DEST_PATH="$BUILD_DIR/MLXR.app"

if [ ! -d "$APP_PATH" ]; then
    echo -e "${RED}  ✗ Built app not found at $APP_PATH${NC}"
    exit 1
fi

# Copy app bundle
echo "  → Copying app bundle..."
rm -rf "$DEST_PATH"
cp -R "$APP_PATH" "$DEST_PATH"

# Create Resources directory structure
RESOURCES_DIR="$DEST_PATH/Contents/Resources"
mkdir -p "$RESOURCES_DIR"/{ui,bin,configs}

# Bundle React UI
echo "  → Bundling React UI..."
if [ -d "$UI_DIR/dist" ]; then
    rm -rf "$RESOURCES_DIR/ui"
    cp -R "$UI_DIR/dist" "$RESOURCES_DIR/ui"
    echo "    ✓ UI bundled ($(du -sh "$RESOURCES_DIR/ui" | cut -f1))"
else
    echo -e "${RED}    ✗ UI dist not found${NC}"
    exit 1
fi

# Bundle daemon binary (if built)
echo "  → Checking for daemon binary..."
if [ -f "$DAEMON_BUILD/test_daemon" ]; then
    cp "$DAEMON_BUILD/test_daemon" "$RESOURCES_DIR/bin/mlxrunnerd"
    chmod +x "$RESOURCES_DIR/bin/mlxrunnerd"
    echo "    ✓ Daemon bundled ($(du -sh "$RESOURCES_DIR/bin/mlxrunnerd" | cut -f1))"
elif [ -f "$DAEMON_BUILD/mlxrunnerd" ]; then
    cp "$DAEMON_BUILD/mlxrunnerd" "$RESOURCES_DIR/bin/mlxrunnerd"
    chmod +x "$RESOURCES_DIR/bin/mlxrunnerd"
    echo "    ✓ Daemon bundled ($(du -sh "$RESOURCES_DIR/bin/mlxrunnerd" | cut -f1))"
else
    echo -e "${YELLOW}    ⚠ Daemon binary not found (will need to be built separately)${NC}"
    echo "    Build with: make mlxr_daemon"
fi

# Bundle configs
echo "  → Bundling configurations..."
cp "$APP_DIR/MLXR/Resources/server.yaml" "$RESOURCES_DIR/configs/server.yaml"
cp "$APP_DIR/MLXR/Resources/com.mlxr.mlxrunnerd.plist" "$RESOURCES_DIR/configs/"
echo "    ✓ Configs bundled"

# Bundle BridgeInjector.js
if [ -f "$APP_DIR/MLXR/Bridge/BridgeInjector.js" ]; then
    cp "$APP_DIR/MLXR/Bridge/BridgeInjector.js" "$RESOURCES_DIR/"
    echo "    ✓ Bridge script bundled"
fi

echo -e "${GREEN}  ✓ All resources bundled${NC}"
echo ""

# ============================================================================
# Step 5: Verify bundle structure
# ============================================================================
echo -e "${YELLOW}[5/6] Verifying bundle...${NC}"

CHECKS=0
TOTAL=0

check_resource() {
    TOTAL=$((TOTAL + 1))
    if [ -e "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $2"
        CHECKS=$((CHECKS + 1))
    else
        echo -e "  ${RED}✗${NC} $2 (missing: $1)"
    fi
}

check_resource "$DEST_PATH/Contents/MacOS/MLXR" "Executable"
check_resource "$RESOURCES_DIR/ui/index.html" "React UI"
check_resource "$RESOURCES_DIR/ui/assets" "UI assets"
check_resource "$RESOURCES_DIR/configs/server.yaml" "Server config"
check_resource "$RESOURCES_DIR/configs/com.mlxr.mlxrunnerd.plist" "Launchd plist"
check_resource "$RESOURCES_DIR/BridgeInjector.js" "Bridge script"

if [ -f "$RESOURCES_DIR/bin/mlxrunnerd" ]; then
    check_resource "$RESOURCES_DIR/bin/mlxrunnerd" "Daemon binary"
fi

echo ""
echo "  Bundle verification: $CHECKS/$TOTAL checks passed"

if [ $CHECKS -lt 6 ]; then
    echo -e "${YELLOW}  ⚠ Some resources are missing (see above)${NC}"
else
    echo -e "${GREEN}  ✓ Bundle is complete${NC}"
fi
echo ""

# ============================================================================
# Step 6: Final summary
# ============================================================================
echo -e "${YELLOW}[6/6] Build summary${NC}"
echo "  → Configuration: $CONFIGURATION"
echo "  → App location: $DEST_PATH"
echo "  → App size: $(du -sh "$DEST_PATH" | cut -f1)"
echo ""

# Get bundle info
if [ -f "$DEST_PATH/Contents/Info.plist" ]; then
    VERSION=$(defaults read "$DEST_PATH/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null || echo "unknown")
    BUILD=$(defaults read "$DEST_PATH/Contents/Info.plist" CFBundleVersion 2>/dev/null || echo "unknown")
    echo "  → Version: $VERSION (build $BUILD)"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Build Complete! ✓             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo "To run the app:"
echo -e "  ${GREEN}open \"$DEST_PATH\"${NC}"
echo ""
echo "To test in development mode with Vite dev server:"
echo -e "  ${GREEN}MLXR_DEV_MODE=1 open \"$DEST_PATH\"${NC}"
echo ""
