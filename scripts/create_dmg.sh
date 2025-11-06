#!/bin/bash
#
# create_dmg.sh
# Create distributable DMG for MLXR.app
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Creating DMG...${NC}"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="${1:-$PROJECT_ROOT/build/macos/MLXR.app}"
DMG_NAME="MLXR-$(sw_vers -productVersion | cut -d. -f1-2)-$(uname -m).dmg"
DMG_PATH="$PROJECT_ROOT/build/$DMG_NAME"

if [ ! -d "$APP_PATH" ]; then
    echo -e "${RED}ERROR: App not found at $APP_PATH${NC}"
    exit 1
fi

# Get version from Info.plist
VERSION=$(defaults read "$APP_PATH/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "1.0.0")
echo "Version: $VERSION"

# Create temporary DMG directory
TMP_DMG_DIR=$(mktemp -d)
trap "rm -rf $TMP_DMG_DIR" EXIT

echo -e "${YELLOW}Preparing DMG contents...${NC}"

# Copy app
cp -R "$APP_PATH" "$TMP_DMG_DIR/"

# Create Applications symlink
ln -s /Applications "$TMP_DMG_DIR/Applications"

# Create README
cat > "$TMP_DMG_DIR/README.txt" << EOF
MLXR - High-Performance LLM Inference for Apple Silicon

Version: $VERSION

Installation:
1. Drag MLXR.app to the Applications folder
2. Open MLXR from Applications
3. The daemon will start automatically

For more information, visit: https://github.com/mlxr/mlxr

Copyright © 2025 MLXR. All rights reserved.
EOF

# Create DMG
echo -e "${YELLOW}Creating DMG...${NC}"
rm -f "$DMG_PATH"

hdiutil create \
    -volname "MLXR $VERSION" \
    -srcfolder "$TMP_DMG_DIR" \
    -ov \
    -format UDZO \
    -fs HFS+ \
    "$DMG_PATH"

echo -e "${GREEN}✓ DMG created: $DMG_PATH${NC}"
echo "Size: $(du -h "$DMG_PATH" | cut -f1)"
