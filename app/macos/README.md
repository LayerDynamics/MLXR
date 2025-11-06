# MLXR macOS App

Native macOS host application for MLXR inference engine.

## Overview

The MLXR macOS app is the native host that:
- Hosts the React UI in a WKWebView
- Provides JavaScript bridge for frontend ↔ daemon communication
- Manages daemon lifecycle via launchd
- Integrates with macOS (tray icon, dock, launch at login)

## Architecture

```
MLXR.app
├── AppDelegate - Main app lifecycle
├── Controllers/
│   ├── TrayController - Menu bar integration
│   ├── MainWindowController - Window management
│   └── WebViewController - WebView hosting
├── Bridge/
│   ├── HostBridge - Bridge protocol
│   ├── MessageHandlers - Bridge implementation
│   ├── UnixSocketClient - Daemon communication
│   └── BridgeInjector.js - JavaScript injection
├── Daemon/
│   ├── DaemonManager - Lifecycle management
│   ├── LaunchdManager - launchctl operations
│   └── HealthMonitor - Status checks
├── Services/
│   ├── KeychainManager - Secure token storage
│   ├── ConfigManager - YAML config handling
│   └── LoginItemManager - Launch at login
└── Views/
    └── TrayPopoverView - Quick status (SwiftUI)
```

## Requirements

- macOS 14.0+ (Sonoma or later)
- Xcode 15+ with Command Line Tools
- Node.js 18+ (for building React UI)
- Apple Silicon (M2, M3, M4)

## Building

### Quick Build

```bash
# From project root
make app
```

### Step-by-Step

```bash
# 1. Build React UI
cd app/ui
npm install
npm run build

# 2. Build macOS app
cd ../..
./scripts/build_app.sh

# 3. Run the app
open build/macos/MLXR.app
```

### Development Mode

To develop with hot reload:

```bash
# Terminal 1: Start React dev server
cd app/ui
npm run dev

# Terminal 2: Build and run app in dev mode
make app-dev
make app-run
```

The app will load from `http://localhost:5173` instead of the bundled files.

## Xcode Project Setup

The Xcode project is defined in `project.yml` and generated using **XcodeGen**. This ensures consistent configuration across developers and makes the project easy to regenerate.

### Automatic Setup (Recommended)

```bash
# From project root
./scripts/setup_xcode_project.sh
```

This script will:
1. Install XcodeGen (via Homebrew) if needed
2. Generate `MLXR.xcodeproj` from `project.yml`
3. Validate project structure and dependencies
4. Show next steps for code signing and building

### Manual Setup

If you prefer to set up manually:

```bash
# Install XcodeGen
brew install xcodegen

# Generate Xcode project
cd app/macos
xcodegen generate
```

### Manual Xcode Configuration

If the Xcode project doesn't exist yet and you prefer manual creation:

1. Open Xcode
2. Create New Project → macOS → App
3. Product Name: **MLXR**
4. Bundle ID: **com.mlxr.app**
5. Language: **Swift**
6. UI: **Storyboard** (we'll use programmatic UI)
7. Save at: `app/macos/`

8. Add all Swift files from `MLXR/` subdirectories:
   - App/*.swift
   - Controllers/*.swift
   - Bridge/*.swift (Swift files only)
   - Daemon/*.swift
   - Services/*.swift
   - Views/*.swift

9. Add BridgeInjector.js:
   - Add to target as resource
   - Build Phases → Copy Bundle Resources

10. Add Resources:
    - Info.plist
    - MLXR.entitlements
    - Assets.xcassets

11. Configure Build Settings:
    - Deployment Target: 14.0
    - Architectures: arm64
    - Code Signing: Developer ID Application (for distribution)

### XcodeGen Configuration

The `project.yml` file defines all project settings:
- **Target**: MLXR (macOS App)
- **Deployment Target**: macOS 14.0+
- **Architectures**: arm64 (Apple Silicon only)
- **Swift Version**: 5.9
- **Frameworks**: Cocoa, WebKit, Security, ServiceManagement, UserNotifications
- **Build Settings**: Hardened Runtime, Code Signing, Entitlements

The configuration includes:
- Pre-build check for React UI
- Post-build scripts to copy daemon binary and config
- Development and Release configurations
- Environment variables for dev mode

To modify project settings, edit `project.yml` and regenerate:
```bash
cd app/macos && xcodegen generate
```

## JavaScript Bridge

The bridge enables communication between React and native code:

### JavaScript Side (Frontend)

```typescript
// Available globally as window.__HOST__
interface HostBridge {
  // HTTP request to daemon
  request(path: string, init?: RequestInit): Promise<Response>

  // File picker
  openPathDialog(type: 'models' | 'cache'): Promise<string | null>

  // Config management
  readConfig(): Promise<string>
  writeConfig(yaml: string): Promise<void>

  // Daemon control
  startDaemon(): Promise<void>
  stopDaemon(): Promise<void>

  // Version info
  getVersion(): Promise<{ app: string; daemon: string }>
}
```

### Swift Side (Native)

Implemented in `MessageHandlers.swift` conforming to `HostBridge` protocol.

### Message Flow

1. **JavaScript → Swift**:
   ```javascript
   window.__HOST__.request('/v1/models', { method: 'GET' })
   ```

2. **Bridge Injector** sends message via WebKit:
   ```javascript
   webkit.messageHandlers.hostBridge.postMessage({
     id: 1,
     method: 'request',
     params: { path: '/v1/models', init: { method: 'GET' } }
   })
   ```

3. **MessageHandlers** receives and processes:
   ```swift
   func userContentController(...) {
     // Parse message
     // Dispatch to appropriate handler
     // Send response back
   }
   ```

4. **Response** sent back to JavaScript:
   ```javascript
   window.handleBridgeResponse(id, error, result)
   ```

## Daemon Communication

The app communicates with the daemon via Unix Domain Socket:

- Socket path: `~/Library/Application Support/MLXRunner/run/mlxrunner.sock`
- Protocol: HTTP over UDS
- Implemented in: `UnixSocketClient.swift`

### Health Checks

The app periodically checks daemon health:

```swift
GET /health → { "status": "ok", "uptime": 123.45, ... }
```

## Daemon Lifecycle

Managed via launchd:

### Plist Location
`~/Library/LaunchAgents/com.mlxr.mlxrunnerd.plist`

### Commands

```bash
# Start daemon
launchctl load -w ~/Library/LaunchAgents/com.mlxr.mlxrunnerd.plist

# Stop daemon
launchctl unload -w ~/Library/LaunchAgents/com.mlxr.mlxrunnerd.plist

# Check status
launchctl list | grep mlxrunnerd
```

### Logs

- stdout: `~/Library/Logs/mlxrunnerd.out.log`
- stderr: `~/Library/Logs/mlxrunnerd.err.log`

## First Run Setup

On first launch, the app:

1. Creates directory structure:
   ```
   ~/Library/Application Support/MLXRunner/
   ├── bin/           # Daemon binary
   ├── models/        # Model files
   ├── cache/         # Cache data
   └── run/           # Socket
   ```

2. Copies daemon binary from bundle
3. Creates default `server.yaml`
4. Installs launchd agent
5. Starts daemon

## Configuration

Server config location:
`~/Library/Application Support/MLXRunner/server.yaml`

Managed via `ConfigManager.swift` with YAML read/write support.

## Code Signing & Distribution

### Sign the App

```bash
# Set signing identity
export MLXR_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"

# Sign
./scripts/sign_app.sh

# Verify
codesign --verify --deep --strict build/macos/MLXR.app
spctl --assess --type execute build/macos/MLXR.app
```

### Create DMG

```bash
./scripts/create_dmg.sh
```

Output: `build/MLXR-<version>.dmg`

### Notarization (Optional)

```bash
# Submit for notarization
xcrun notarytool submit build/MLXR-*.dmg \
  --apple-id "your@email.com" \
  --team-id "TEAMID" \
  --password "@keychain:AC_PASSWORD"

# Wait for approval, then staple
xcrun stapler staple build/MLXR.app
```

## Troubleshooting

### WebView Not Loading

**Symptom**: Blank window or error page

**Solutions**:
1. Check if UI is built: `ls app/ui/dist/index.html`
2. Build UI: `cd app/ui && npm run build`
3. Check symlink: `ls -la app/macos/MLXR/Resources/ui`

### Daemon Won't Start

**Symptom**: "Failed to start daemon" error

**Solutions**:
1. Check if binary exists: `ls ~/Library/Application Support/MLXRunner/bin/mlxrunnerd`
2. Check permissions: `ls -l ~/Library/Application Support/MLXRunner/bin/mlxrunnerd` (should be 755)
3. Check logs: `tail -f ~/Library/Logs/mlxrunnerd.err.log`
4. Try manual start: `~/Library/Application Support/MLXRunner/bin/mlxrunnerd --config ~/Library/Application Support/MLXRunner/server.yaml`

### Bridge Not Working

**Symptom**: Frontend can't communicate with daemon

**Solutions**:
1. Check if BridgeInjector.js is included in bundle
2. Open Web Inspector: Enable in Develop menu, right-click webview → Inspect Element
3. Check console for bridge errors
4. Verify socket exists: `ls -la ~/Library/Application Support/MLXRunner/run/mlxrunner.sock`

## Development Tips

### Enable Web Inspector

In `WebViewController.swift`:
```swift
if isDevelopment {
    config.preferences.setValue(true, forKey: "developerExtrasEnabled")
}
```

Then: Right-click in WebView → Inspect Element

### Hot Reload

Use development mode for React hot reload:
```bash
# Start dev server
cd app/ui && npm run dev

# Run app with MLXR_DEV_MODE=1
make app-dev
```

### Debug Daemon Communication

Add logging in `UnixSocketClient.swift`:
```swift
print("[UDS] Sending: \(httpRequest)")
print("[UDS] Received: \(responseString)")
```

## Notification System

The app uses the modern UserNotifications framework to keep users informed:

### Notification Types

1. **Daemon Status**
   - Started: "MLXR Daemon Started"
   - Stopped: "MLXR Daemon Stopped"
   - Crashed: "MLXR Daemon Crashed" (with restart action)

2. **Model Events**
   - Model loaded: "Model Loaded - [model name]"
   - Model unloaded: "Model Unloaded - [model name]"

3. **Errors**
   - Critical errors with actionable information
   - Non-critical warnings

4. **Updates**
   - "Update Available - MLXR [version]" (with install action)

### Notification Actions

Users can interact with notifications:
- **Daemon Crashed**: Restart button
- **Update Available**: Install Update button
- **Any Notification**: Tap to open main window

### Implementation

```swift
// Daemon started
NotificationManager.shared.notifyDaemonStarted()

// Model loaded
NotificationManager.shared.notifyModelLoaded(modelName: "Llama-3-8B")

// Error
NotificationManager.shared.notifyError(
    title: "Configuration Error",
    message: "Invalid server.yaml",
    critical: true
)

// Update available
NotificationManager.shared.notifyUpdateAvailable(
    version: "1.1.0",
    releaseNotes: "Bug fixes and performance improvements"
)
```

### Permission Handling

- Requested on first launch
- User can enable/disable in System Settings
- Gracefully handles denied permissions

## Auto-Update System

The app includes a complete UpdateManager ready for Sparkle framework integration.

### UpdateManager Features

**Automatic Updates:**
- Periodic background checks (every hour)
- Silent update detection
- User notifications when updates available
- Manual update checks via menu

**Configuration:**
```swift
// Enable automatic checks
UpdateManager.shared.setAutomaticUpdateChecksEnabled(true)

// Disable automatic downloads (user confirmation required)
UpdateManager.shared.setAutomaticDownloadEnabled(false)

// Manual check
UpdateManager.shared.checkForUpdates()

// Get current version
let version = UpdateManager.shared.getCurrentVersion()
```

### Integrating Sparkle Framework

**Step 1: Add Sparkle to Xcode Project**

```bash
# Download Sparkle 2.x
curl -L https://github.com/sparkle-project/Sparkle/releases/download/2.5.0/Sparkle-2.5.0.tar.xz | tar xJ

# Add Sparkle.framework to Xcode project
# Drag Sparkle.framework into Frameworks folder
# Embed & Sign in Build Phases
```

**Step 2: Generate EdDSA Signing Keys**

```bash
# Generate private key (keep secure!)
./Sparkle.framework/Resources/generate_keys

# Output:
# Public key: <public-key-string>
# Private key: <private-key-string>
```

**Step 3: Update Info.plist**

The app is already configured with:
- `SUFeedURL`: Update appcast URL
- `SUPublicEDKey`: Add your public key here
- `SUEnableAutomaticChecks`: true

**Step 4: Create Appcast**

```xml
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>MLXR Updates</title>
    <item>
      <title>Version 1.1.0</title>
      <sparkle:version>1.1.0</sparkle:version>
      <sparkle:releaseNotesLink>
        https://mlxr.dev/releases/1.1.0.html
      </sparkle:releaseNotesLink>
      <pubDate>Wed, 06 Nov 2025 12:00:00 +0000</pubDate>
      <enclosure url="https://mlxr.dev/downloads/MLXR-1.1.0.dmg"
                 sparkle:version="1.1.0"
                 sparkle:edSignature="<signature>"
                 length="50000000"
                 type="application/octet-stream" />
    </item>
  </channel>
</rss>
```

**Step 5: Sign Releases**

```bash
# Sign DMG with Sparkle
./Sparkle.framework/Resources/sign_update MLXR-1.1.0.dmg <private-key>

# Output: EdDSA signature for appcast
```

**Step 6: Host Appcast**

- Upload appcast.xml to HTTPS server
- Update `SUFeedURL` in Info.plist
- Ensure server allows HTTP HEAD requests

### Update Flow

1. **Background Check**: Every hour, UpdateManager checks appcast
2. **Update Found**: User receives notification with version info
3. **User Action**: Click "Install Update" or use menu
4. **Download**: Sparkle downloads and verifies DMG
5. **Install**: User confirms, app quits and updates
6. **Relaunch**: Updated app launches automatically

### Testing Updates

**Local Testing:**
```bash
# Serve appcast locally
python3 -m http.server 8080

# Update Info.plist temporarily
SUFeedURL: http://localhost:8080/appcast.xml

# Test with lower version number
# Sparkle will detect "update" available
```

### UpdateManager API

```swift
// Check for updates manually
UpdateManager.shared.checkForUpdates()

// Check silently in background
UpdateManager.shared.checkForUpdatesInBackground()

// Enable/disable automatic checks
UpdateManager.shared.setAutomaticUpdateChecksEnabled(enabled)

// Query update state
if UpdateManager.shared.isUpdateAvailable {
    print("Version \(UpdateManager.shared.latestVersion!) available")
}
```

### Conditional Compilation

UpdateManager uses conditional compilation to work without Sparkle:

```swift
#if canImport(Sparkle)
// Full Sparkle functionality
#else
// Graceful fallback - logs warning
#endif
```

This means the app compiles and runs without Sparkle, but auto-updates are disabled.

## Testing

The macOS app includes comprehensive automated and manual tests. See [TESTING.md](TESTING.md) for the complete testing guide.

### Quick Start

**Run all tests:**
```bash
# Using Xcode
open app/macos/MLXR.xcodeproj
# Press Cmd+U

# Using command line
xcodebuild test \
  -project app/macos/MLXR.xcodeproj \
  -scheme MLXR \
  -destination 'platform=macOS,arch=arm64'
```

### Test Suites

1. **BridgeTests** (18 tests)
   - JavaScript bridge communication
   - Message handling and validation
   - Bridge method availability
   - Error handling and performance

2. **DaemonManagerTests** (12 tests)
   - Daemon lifecycle management
   - Path configuration
   - Health monitoring
   - Concurrent access safety

3. **ServicesTests** (25 tests)
   - KeychainManager: Token generation, save/retrieve, rotation
   - ConfigManager: Read/write, validation, default config
   - LoginItemManager: Status and enable/disable operations

4. **IntegrationTests** (10 tests)
   - First run setup flow
   - Bridge + Daemon integration
   - Config + Keychain integration
   - End-to-end workflow scenarios
   - Error recovery

### Manual Testing

Interactive tests that require user interaction:
- **Bridge Communication**: Test with React UI and browser console
- **Daemon Start/Stop/Restart**: Verify lifecycle management
- **File Picker Integration**: Test model/folder selection dialogs
- **Launch at Login**: Verify SMAppService integration

See [TESTING.md](TESTING.md) for detailed manual testing procedures.

### Test Coverage

Current coverage: **~85% of core functionality**
- Bridge: 90%
- Daemon Management: 85%
- Services: 95%
- Integration: 75%

## Future Enhancements

- [ ] App icon and tray icon design
- [ ] More comprehensive logging
- [ ] Delta updates for faster downloads
- [ ] UI automation tests (XCUITest)
- [ ] Snapshot tests for UI regression testing

## Related Documentation

- [Frontend Implementation](../ui/README.md) - React UI
- [Build Scripts](../../scripts/) - Automation scripts
- [Main Project](../../README.md) - Overall project docs

## License

Copyright © 2025 MLXR. All rights reserved.
