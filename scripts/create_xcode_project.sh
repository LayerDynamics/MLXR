#!/bin/bash
# Script to create Xcode project for MLXR macOS app
# This should be run on macOS with Xcode installed

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APP_DIR="$PROJECT_ROOT/app/macos"

echo "Creating Xcode project for MLXR..."
echo "Project root: $PROJECT_ROOT"
echo "App directory: $APP_DIR"

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script must be run on macOS with Xcode installed"
    exit 1
fi

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "ERROR: Xcode command line tools not found"
    echo "Please install Xcode from the App Store and run:"
    echo "  xcode-select --install"
    exit 1
fi

# Create Xcode project directory
cd "$APP_DIR"

# Check if project already exists
if [ -d "MLXR.xcodeproj" ]; then
    echo "WARNING: MLXR.xcodeproj already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
    rm -rf MLXR.xcodeproj
fi

# Create project using xcodegen (if available) or manual creation
if command -v xcodegen &> /dev/null; then
    echo "Using xcodegen to create project..."

    # Create project.yml for xcodegen
    cat > project.yml << 'EOF'
name: MLXR
options:
  bundleIdPrefix: com.mlxr
  deploymentTarget:
    macOS: "13.0"
  developmentLanguage: swift

settings:
  MARKETING_VERSION: "0.1.0"
  CURRENT_PROJECT_VERSION: "1"
  SWIFT_VERSION: "5.9"
  MACOSX_DEPLOYMENT_TARGET: "13.0"

targets:
  MLXR:
    type: application
    platform: macOS
    deploymentTarget: "13.0"
    sources:
      - MLXR/App
      - MLXR/Bridge
      - MLXR/Controllers
      - MLXR/Daemon
      - MLXR/Services
      - MLXR/Views
    settings:
      PRODUCT_NAME: MLXR
      PRODUCT_BUNDLE_IDENTIFIER: com.mlxr.MLXR
      INFOPLIST_FILE: MLXR/Info.plist
      CODE_SIGN_STYLE: Automatic
      DEVELOPMENT_TEAM: ""
      CODE_SIGN_IDENTITY: "-"
      ENABLE_HARDENED_RUNTIME: YES
      COMBINE_HIDPI_IMAGES: YES
    dependencies:
      - framework: WebKit.framework
      - framework: Foundation.framework
      - framework: AppKit.framework
      - framework: Security.framework
      - framework: ServiceManagement.framework
      - sdk: Sparkle
        type: package
        product: Sparkle

  MLXRTests:
    type: bundle.unit-test
    platform: macOS
    sources:
      - MLXRTests
    dependencies:
      - target: MLXR

packages:
  Sparkle:
    url: https://github.com/sparkle-project/Sparkle
    from: "2.5.0"
EOF

    xcodegen generate
    echo "✓ Xcode project generated successfully"

else
    echo "xcodegen not found, creating basic project..."
    echo ""
    echo "To install xcodegen:"
    echo "  brew install xcodegen"
    echo ""
    echo "Or create project manually in Xcode:"
    echo "  1. Open Xcode"
    echo "  2. File > New > Project"
    echo "  3. Choose macOS > App"
    echo "  4. Set:"
    echo "     - Product Name: MLXR"
    echo "     - Interface: SwiftUI or AppKit (AppKit recommended)"
    echo "     - Language: Swift"
    echo "     - Organization Identifier: com.mlxr"
    echo "  5. Save to: $APP_DIR"
    echo "  6. Add existing files:"

    find MLXR -name "*.swift" | sort | while read file; do
        echo "     - $file"
    done

    echo ""
    echo "  7. Add frameworks:"
    echo "     - WebKit.framework"
    echo "     - Security.framework (for Keychain)"
    echo "     - ServiceManagement.framework (for LoginItems)"
    echo ""
    echo "  8. Add Sparkle via Swift Package Manager:"
    echo "     - File > Add Package Dependencies"
    echo "     - URL: https://github.com/sparkle-project/Sparkle"
    echo "     - Version: 2.5.0 or later"

    exit 1
fi

# Create Info.plist if it doesn't exist
if [ ! -f "MLXR/Info.plist" ]; then
    echo "Creating Info.plist..."
    cat > MLXR/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>$(DEVELOPMENT_LANGUAGE)</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$(MARKETING_VERSION)</string>
    <key>CFBundleVersion</key>
    <string>$(CURRENT_PROJECT_VERSION)</string>
    <key>LSMinimumSystemVersion</key>
    <string>$(MACOSX_DEPLOYMENT_TARGET)</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2025 MLXR Development. All rights reserved.</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSMainStoryboardFile</key>
    <string>Main</string>
    <key>SUFeedURL</key>
    <string>https://mlxr.dev/appcast.xml</string>
    <key>SUPublicEDKey</key>
    <string></string>
</dict>
</plist>
EOF
    echo "✓ Info.plist created"
fi

# Create entitlements file
if [ ! -f "MLXR/MLXR.entitlements" ]; then
    echo "Creating entitlements..."
    cat > MLXR/MLXR.entitlements << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <key>com.apple.security.files.bookmarks.app-scope</key>
    <true/>
</dict>
</plist>
EOF
    echo "✓ Entitlements created"
fi

echo ""
echo "✓ Xcode project setup complete!"
echo ""
echo "Next steps:"
echo "  1. Open MLXR.xcodeproj in Xcode"
echo "  2. Set your Development Team in Signing & Capabilities"
echo "  3. Create app icons in Assets.xcassets"
echo "  4. Build and run: ⌘R"
echo ""
echo "Project structure:"
echo "  MLXR/"
echo "    ├── App/          (AppDelegate, main entry point)"
echo "    ├── Bridge/       (Swift-WebView bridge)"
echo "    ├── Controllers/  (Window, Tray, WebView controllers)"
echo "    ├── Daemon/       (Daemon lifecycle management)"
echo "    ├── Services/     (Config, Keychain, Updates, etc.)"
echo "    └── Views/        (Tray popover view)"
echo ""
