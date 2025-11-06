#!/bin/bash
#
# sign_app.sh
# Code sign MLXR.app with Developer ID
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Signing MLXR.app...${NC}"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="${1:-$PROJECT_ROOT/build/macos/MLXR.app}"
IDENTITY="${MLXR_SIGN_IDENTITY:-Developer ID Application}"

if [ ! -d "$APP_PATH" ]; then
    echo -e "${RED}ERROR: App not found at $APP_PATH${NC}"
    echo "Please build the app first: ./scripts/build_app.sh"
    exit 1
fi

echo "App path: $APP_PATH"
echo "Identity: $IDENTITY"

# Step 1: Sign frameworks and embedded binaries
echo -e "${YELLOW}Signing embedded binaries...${NC}"
find "$APP_PATH/Contents/Frameworks" -name "*.framework" -o -name "*.dylib" 2>/dev/null | while read framework; do
    echo "  Signing: $(basename "$framework")"
    codesign --force --sign "$IDENTITY" --timestamp --options runtime "$framework" || true
done

# Step 2: Sign the app bundle
echo -e "${YELLOW}Signing app bundle...${NC}"
codesign --force \
    --sign "$IDENTITY" \
    --timestamp \
    --options runtime \
    --entitlements "$PROJECT_ROOT/app/macos/MLXR/Resources/MLXR.entitlements" \
    --deep \
    "$APP_PATH"

# Step 3: Verify signature
echo -e "${YELLOW}Verifying signature...${NC}"
codesign --verify --deep --strict --verbose=2 "$APP_PATH"

# Step 4: Check Gatekeeper
echo -e "${YELLOW}Checking Gatekeeper assessment...${NC}"
spctl --assess --type execute --verbose=4 "$APP_PATH" || echo -e "${YELLOW}Note: Gatekeeper assessment may fail until notarized${NC}"

echo -e "${GREEN}✓ Signing complete!${NC}"
