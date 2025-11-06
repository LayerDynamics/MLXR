#!/bin/bash
#
# build_app.sh
# Build MLXR.app for macOS
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MLXR.app...${NC}"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$PROJECT_ROOT/app/macos"
UI_DIR="$PROJECT_ROOT/app/ui"
BUILD_DIR="$PROJECT_ROOT/build/macos"
SCHEME="MLXR"
CONFIGURATION="${1:-Release}"  # Default to Release, can override with arg

# Step 1: Build React UI
echo -e "${YELLOW}Step 1: Building React UI...${NC}"
cd "$UI_DIR"
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run build
echo -e "${GREEN}✓ React UI built${NC}"

# Step 2: Check if Xcode project exists
if [ ! -d "$APP_DIR/MLXR.xcodeproj" ]; then
    echo -e "${RED}ERROR: Xcode project not found at $APP_DIR/MLXR.xcodeproj${NC}"
    echo "Please create the Xcode project first by opening Xcode and creating a new macOS app project."
    echo "Then add all the Swift/ObjC files from $APP_DIR/MLXR/"
    exit 1
fi

# Step 3: Build macOS app with xcodebuild
echo -e "${YELLOW}Step 2: Building macOS app...${NC}"
cd "$APP_DIR"

xcodebuild \
    -project MLXR.xcodeproj \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$BUILD_DIR" \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    build

echo -e "${GREEN}✓ macOS app built${NC}"

# Step 4: Copy app to build directory
APP_PATH="$BUILD_DIR/Build/Products/$CONFIGURATION/MLXR.app"
DEST_PATH="$BUILD_DIR/MLXR.app"

if [ -d "$APP_PATH" ]; then
    rm -rf "$DEST_PATH"
    cp -R "$APP_PATH" "$DEST_PATH"
    echo -e "${GREEN}✓ App copied to $DEST_PATH${NC}"
else
    echo -e "${RED}ERROR: Built app not found at $APP_PATH${NC}"
    exit 1
fi

# Step 5: Verify build
echo -e "${YELLOW}Verifying build...${NC}"
if [ -d "$DEST_PATH/Contents/Resources/ui" ]; then
    echo -e "${GREEN}✓ UI resources found${NC}"
else
    echo -e "${YELLOW}⚠ UI resources not found (may need manual symlink)${NC}"
fi

echo -e "${GREEN}✓ Build complete!${NC}"
echo "App location: $DEST_PATH"
echo ""
echo "To run the app:"
echo "  open $DEST_PATH"
