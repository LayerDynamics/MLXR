# GitHub Actions Integration Guide

This document lists all GitHub Actions used in MLXR workflows and available alternatives.

## Currently Used Actions

### 1. **XcodeGen Action** (release.yml)
- **Action**: `xavierLowmiller/xcodegen-action@1.1.2`
- **Purpose**: Generates Xcode project from `project.yml` specification
- **Configuration**:
  ```yaml
  - name: Generate Xcode project
    uses: xavierLowmiller/xcodegen-action@1.1.2
    with:
      spec: app/macos/project.yml
      version: '2.38.0'
  ```
- **Benefits**: Faster than `brew install xcodegen`, cached binary
- **Alternatives**: Manual `brew install xcodegen && xcodegen generate`

### 2. **Setup Protoc** (grpc-server-tests.yml)
- **Action**: `arduino/setup-protoc@v3`
- **Purpose**: Installs protobuf compiler for code generation
- **Configuration**:
  ```yaml
  - name: Setup Protobuf Compiler
    uses: arduino/setup-protoc@v3
    with:
      version: '25.x'
      repo-token: ${{ secrets.GITHUB_TOKEN }}
  ```
- **Benefits**: Cross-platform, faster than brew, versioned
- **Alternatives**: `brew install protobuf` (macOS only)

### 3. **Create DMG** (release.yml)
- **Action**: `QQxiaoming/create-dmg-action@v0.0.2`
- **Purpose**: Creates macOS DMG installer
- **Configuration**:
  ```yaml
  - name: Create DMG
    uses: QQxiaoming/create-dmg-action@v0.0.2
    with:
      name: MLXR-${{ steps.version.outputs.VERSION }}-macOS-arm64
      srcdir: build/macos/MLXR.app
  ```
- **Benefits**: Automated DMG creation with proper formatting
- **Alternatives**: Manual `hdiutil create`

---

## Available Actions (Not Yet Used)

### Code Signing & Notarization

#### **Option 1: indygreg/apple-code-sign-action** (Recommended)
- **Best for**: Cross-platform CI (works on Linux/Windows/macOS)
- **Features**:
  - Signs with Developer ID certificate
  - Notarizes with Apple
  - Staples notarization ticket
  - Uses open-source `rcodesign` tool
- **Usage**:
  ```yaml
  - name: Sign and Notarize
    uses: indygreg/apple-code-sign-action@v1
    with:
      input_path: build/macos/MLXR.app
      p12_file: ${{ secrets.APPLE_DEVELOPER_CERTIFICATE_P12_BASE64 }}
      p12_password: ${{ secrets.APPLE_DEVELOPER_CERTIFICATE_PASSWORD }}
      apple_id: ${{ secrets.APPLE_ID }}
      apple_id_password: ${{ secrets.APPLE_ID_PASSWORD }}
      team_id: ${{ secrets.APPLE_TEAM_ID }}
      notarize: true
      staple: true
  ```
- **Required Secrets**:
  - `APPLE_DEVELOPER_CERTIFICATE_P12_BASE64` - Base64-encoded .p12 cert
  - `APPLE_DEVELOPER_CERTIFICATE_PASSWORD` - Certificate password
  - `APPLE_ID` - Apple Developer email
  - `APPLE_ID_PASSWORD` - App-specific password (not main password!)
  - `APPLE_TEAM_ID` - 10-character team ID

#### **Option 2: toitlang/action-macos-sign-notarize**
- **Best for**: macOS-only workflows
- **Features**: Native macOS tools (codesign, notarytool)
- **Limitation**: Requires macOS runner
- **Usage**:
  ```yaml
  - uses: toitlang/action-macos-sign-notarize@v1
    with:
      certificate: ${{ secrets.MACOS_CERTIFICATE }}
      certificate-password: ${{ secrets.MACOS_CERTIFICATE_PWD }}
      apple-id: ${{ secrets.APPLE_ID }}
      apple-id-password: ${{ secrets.APPLE_ID_PASSWORD }}
  ```

#### **Option 3: Manual with notarytool** (Apple's official tool)
- **Best for**: Full control, latest Apple APIs
- **Usage**:
  ```yaml
  - name: Notarize
    run: |
      xcrun notarytool submit MLXR.zip \
        --apple-id ${{ secrets.APPLE_ID }} \
        --team-id ${{ secrets.APPLE_TEAM_ID }} \
        --password ${{ secrets.APPLE_ID_PASSWORD }} \
        --wait

      xcrun stapler staple MLXR.app
  ```

### gRPC/Protobuf Tools

#### **setup-grpc** (for Linux runners)
- **Action**: `eWaterCycle/setup-grpc@v5`
- **Purpose**: Installs gRPC C++ on Ubuntu runners
- **Note**: Not needed for macOS (use brew instead)
- **Usage**:
  ```yaml
  - uses: eWaterCycle/setup-grpc@v5
    with:
      grpc-version: '1.51.1'
  ```

---

## Recommended Setup

### For Development Builds (Current)
```yaml
# XcodeGen
- uses: xavierLowmiller/xcodegen-action@1.1.2
  with:
    spec: app/macos/project.yml

# Protobuf
- uses: arduino/setup-protoc@v3
  with:
    version: '25.x'

# Ad-hoc signing (local testing)
- run: codesign --force --deep --sign - MLXR.app
```

### For Production Releases
```yaml
# XcodeGen
- uses: xavierLowmiller/xcodegen-action@1.1.2

# Protobuf
- uses: arduino/setup-protoc@v3

# Sign & Notarize (cross-platform)
- uses: indygreg/apple-code-sign-action@v1
  with:
    input_path: MLXR.app
    notarize: true
    staple: true
  # Requires APPLE_* secrets

# Create DMG
- uses: QQxiaoming/create-dmg-action@v0.0.2
```

---

## Setting Up Production Signing

### 1. Generate Certificates (requires Apple Developer Program - $99/year)

**On macOS with Xcode:**
```bash
# Create Certificate Signing Request
# Xcode > Settings > Accounts > Manage Certificates > +

# Download certificates from developer.apple.com:
# - Developer ID Application (for app signing)
# - Developer ID Installer (for PKG signing)
```

### 2. Export Certificate as P12

```bash
# In Keychain Access
# Right-click certificate > Export "Developer ID Application"
# Save as .p12 with password
# Convert to base64:
base64 -i certificate.p12 | pbcopy
```

### 3. Create App-Specific Password

1. Go to https://appleid.apple.com
2. Sign In
3. Security > App-Specific Passwords
4. Generate password for "MLXR CI/CD"
5. Save password (won't be shown again!)

### 4. Add GitHub Secrets

In repository Settings > Secrets and variables > Actions:

- `APPLE_DEVELOPER_CERTIFICATE_P12_BASE64`: [paste base64 from step 2]
- `APPLE_DEVELOPER_CERTIFICATE_PASSWORD`: [p12 password]
- `APPLE_ID`: your@email.com
- `APPLE_ID_PASSWORD`: [app-specific password from step 3]
- `APPLE_TEAM_ID`: Find at developer.apple.com/account (10 chars)

### 5. Enable Signing in Workflow

Uncomment the signing section in `.github/workflows/release.yml`:

```yaml
# Replace this:
- name: Sign application (ad-hoc)
  run: codesign --force --deep --sign - "$APP_BUNDLE"

# With this:
- name: Sign and Notarize
  uses: indygreg/apple-code-sign-action@v1
  with:
    input_path: build/macos/MLXR.app
    p12_file: ${{ secrets.APPLE_DEVELOPER_CERTIFICATE_P12_BASE64 }}
    p12_password: ${{ secrets.APPLE_DEVELOPER_CERTIFICATE_PASSWORD }}
    apple_id: ${{ secrets.APPLE_ID }}
    apple_id_password: ${{ secrets.APPLE_ID_PASSWORD }}
    team_id: ${{ secrets.APPLE_TEAM_ID }}
    notarize: true
    staple: true
```

---

## Testing Workflows Locally

### Act (Run GitHub Actions locally)

```bash
# Install act
brew install act

# Test workflow
act -W .github/workflows/release.yml

# With secrets
act -W .github/workflows/release.yml --secret-file .secrets
```

---

## Workflow Performance Tips

### 1. Use Caching
All workflows already use caching for:
- ccache (C++ compilation)
- Homebrew packages
- Node modules
- Metal kernels

### 2. Use Artifacts
- Upload build artifacts for debugging
- Download artifacts in dependent jobs
- Set retention-days appropriately (7 for CI, 90 for releases)

### 3. Use Concurrency Control
All workflows have:
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

This cancels old runs when new commits are pushed.

---

## References

- [XcodeGen Action](https://github.com/marketplace/actions/xcodegen)
- [Setup Protoc](https://github.com/marketplace/actions/setup-protoc)
- [Apple Code Sign Action](https://github.com/indygreg/apple-code-sign-action)
- [Create DMG Action](https://github.com/QQxiaoming/create-dmg-action)
- [Apple Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [GitHub Actions - macOS](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)

---

## Current Workflow Status

| Workflow | XcodeGen | Protobuf | gRPC | Signing | Notarization | Status |
|----------|----------|----------|------|---------|--------------|--------|
| ci.yml | N/A | N/A | N/A | N/A | N/A | ✅ Working |
| release.yml | ✅ Action | Manual | N/A | Ad-hoc | ❌ Not configured | ⚠️ Dev only |
| grpc-server-tests.yml | N/A | ✅ Action | brew | N/A | N/A | ✅ Working |
| daemon-test.yml | N/A | N/A | N/A | N/A | N/A | ✅ Working |

**Next Steps for Production:**
1. Obtain Apple Developer Program membership ($99/year)
2. Generate certificates and app-specific password
3. Add secrets to GitHub repository
4. Enable `indygreg/apple-code-sign-action` in release.yml
5. Test with workflow_dispatch trigger
