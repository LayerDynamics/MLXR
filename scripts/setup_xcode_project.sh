#!/bin/bash
#
# setup_xcode_project.sh
#
# Generates MLXR.xcodeproj using XcodeGen from project.yml specification.
# This ensures consistent Xcode project configuration across developers.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACOS_DIR="${PROJECT_ROOT}/app/macos"
PROJECT_YML="${MACOS_DIR}/project.yml"
XCODEPROJ="${MACOS_DIR}/MLXR.xcodeproj"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}MLXR Xcode Project Setup${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check if xcodegen is installed
if ! command -v xcodegen &> /dev/null; then
    echo -e "${YELLOW}XcodeGen not found. Installing via Homebrew...${NC}"

    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Error: Homebrew not found.${NC}"
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    fi

    brew install xcodegen
fi

# Check if project.yml exists
if [ ! -f "$PROJECT_YML" ]; then
    echo -e "${RED}Error: project.yml not found at $PROJECT_YML${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found project.yml"
echo -e "${GREEN}✓${NC} XcodeGen installed ($(xcodegen --version))"
echo ""

# Backup existing project if it exists
if [ -d "$XCODEPROJ" ]; then
    echo -e "${YELLOW}Existing Xcode project found. Creating backup...${NC}"
    BACKUP="${XCODEPROJ}.backup.$(date +%Y%m%d_%H%M%S)"
    mv "$XCODEPROJ" "$BACKUP"
    echo -e "${GREEN}✓${NC} Backed up to $(basename "$BACKUP")"
    echo ""
fi

# Generate Xcode project
echo -e "${BLUE}Generating Xcode project...${NC}"
cd "$MACOS_DIR"
xcodegen generate

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓${NC} Xcode project generated successfully!"
    echo ""
    echo -e "${BLUE}Project location:${NC} $XCODEPROJ"
else
    echo -e "${RED}Error: Failed to generate Xcode project${NC}"
    exit 1
fi

# Validate project structure
echo ""
echo -e "${BLUE}Validating project structure...${NC}"

# Check for required source files
REQUIRED_FILES=(
    "MLXR/App/AppDelegate.swift"
    "MLXR/Controllers/TrayController.swift"
    "MLXR/Controllers/MainWindowController.swift"
    "MLXR/Controllers/WebViewController.swift"
    "MLXR/Bridge/HostBridge.swift"
    "MLXR/Bridge/MessageHandlers.swift"
    "MLXR/Bridge/UnixSocketClient.swift"
    "MLXR/Bridge/BridgeInjector.js"
    "MLXR/Daemon/DaemonManager.swift"
    "MLXR/Daemon/LaunchdManager.swift"
    "MLXR/Daemon/HealthMonitor.swift"
    "MLXR/Services/KeychainManager.swift"
    "MLXR/Services/ConfigManager.swift"
    "MLXR/Services/LoginItemManager.swift"
    "MLXR/Services/NotificationManager.swift"
    "MLXR/Services/UpdateManager.swift"
    "MLXR/Views/TrayPopoverView.swift"
    "MLXR/Resources/Info.plist"
    "MLXR/Resources/MLXR.entitlements"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${MACOS_DIR}/${file}" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All required source files present"
else
    echo -e "${YELLOW}⚠${NC}  Missing ${#MISSING_FILES[@]} file(s):"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "   ${YELLOW}•${NC} $file"
    done
fi

# Check for React UI build
if [ -d "${MACOS_DIR}/MLXR/Resources/ui" ]; then
    echo -e "${GREEN}✓${NC} React UI build found"
else
    echo -e "${YELLOW}⚠${NC}  React UI not built yet"
    echo "   Run: cd app/ui && npm install && npm run build"
fi

# Check for daemon binary
DAEMON_BIN="${PROJECT_ROOT}/build/cmake/bin/mlxrunnerd"
if [ -f "$DAEMON_BIN" ]; then
    echo -e "${GREEN}✓${NC} Daemon binary found"
else
    echo -e "${YELLOW}⚠${NC}  Daemon binary not built yet"
    echo "   Run: make build"
fi

# Print next steps
echo ""
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Next Steps${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""
echo "1. Open Xcode project:"
echo "   open $XCODEPROJ"
echo ""
echo "2. Configure code signing:"
echo "   - Select MLXR target"
echo "   - Go to Signing & Capabilities"
echo "   - Set your Development Team"
echo "   - Select 'Developer ID Application' certificate"
echo ""
echo "3. Build dependencies (if not done):"
echo "   - React UI:  cd app/ui && npm install && npm run build"
echo "   - Daemon:    make build"
echo ""
echo "4. Build and run:"
echo "   - Press Cmd+R in Xcode, or"
echo "   - Run: xcodebuild -project $XCODEPROJ -scheme MLXR -configuration Debug"
echo ""
echo -e "${GREEN}Setup complete!${NC}"
