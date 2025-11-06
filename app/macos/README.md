# MLXR macOS App

Native macOS application for MLXR - tray/dock UI with WebView for the React frontend.

## Project Structure

```
app/macos/
├── MLXR/
│   ├── App/                    # Application entry point
│   │   └── AppDelegate.swift   # Main app delegate
│   ├── Bridge/                 # Swift-JavaScript bridge
│   │   ├── HostBridge.swift    # Main bridge interface
│   │   ├── MessageHandlers.swift  # JS message handlers
│   │   └── UnixSocketClient.swift # UDS client for daemon
│   ├── Controllers/            # View controllers
│   │   ├── MainWindowController.swift  # Main window
│   │   ├── TrayController.swift        # Menu bar tray
│   │   └── WebViewController.swift     # WebView container
│   ├── Daemon/                 # Daemon lifecycle management
│   │   ├── DaemonManager.swift      # Daemon start/stop/monitor
│   │   ├── HealthMonitor.swift      # Health check polling
│   │   └── LaunchdManager.swift     # launchd integration
│   ├── Services/               # App services
│   │   ├── ConfigManager.swift      # Config read/write
│   │   ├── KeychainManager.swift    # Secure token storage
│   │   ├── LoginItemManager.swift   # Launch at login
│   │   ├── NotificationManager.swift # User notifications
│   │   └── UpdateManager.swift      # Sparkle auto-updates
│   ├── Views/                  # SwiftUI/AppKit views
│   │   └── TrayPopoverView.swift    # Tray popover
│   ├── Info.plist              # App metadata
│   └── MLXR.entitlements       # Sandbox entitlements
├── MLXRTests/                  # Unit tests
│   ├── BridgeTests.swift
│   ├── DaemonManagerTests.swift
│   ├── IntegrationTests.swift
│   └── ServicesTests.swift
├── project.yml                 # XcodeGen specification
└── README.md                   # This file
```

## Requirements

- **macOS**: 13.0 (Ventura) or later
- **Xcode**: 15.0 or later
- **Swift**: 5.9 or later

## Building the Project

### Option 1: Using XcodeGen (Recommended)

1. Install XcodeGen:
   \`\`\`bash
   brew install xcodegen
   \`\`\`

2. Generate Xcode project:
   \`\`\`bash
   cd app/macos
   xcodegen generate
   \`\`\`

3. Open in Xcode:
   \`\`\`bash
   open MLXR.xcodeproj
   \`\`\`

4. Set your Development Team in "Signing & Capabilities"

5. Build and run: \`⌘R\`

### Option 2: Using the Setup Script

From the project root:

\`\`\`bash
./scripts/create_xcode_project.sh
\`\`\`

This script will:
- Check for macOS and Xcode
- Generate the project using xcodegen (or provide manual instructions)
- Create Info.plist and entitlements if missing
- Set up project structure

## Testing

Run tests in Xcode with \`⌘U\` or:

\`\`\`bash
xcodebuild test -scheme MLXR -destination 'platform=macOS'
\`\`\`

## License

Copyright © 2025 MLXR Development. All rights reserved.
