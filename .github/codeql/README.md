# CodeQL Configuration for MLXR

This directory contains CodeQL security analysis configuration.

## Current Setup

MLXR uses **GitHub's default CodeQL setup** for security scanning. The configuration in `codeql-config.yml` excludes paths that aren't ready for analysis yet (like the incomplete macOS Xcode project).

## Configuration

The default CodeQL setup is enabled in the repository settings and analyzes:
- **JavaScript/TypeScript**: `app/ui/src/**`
- **Python**: Python scripts and tooling
- **C++**: `core/**`, `daemon/**`, `tests/**`, `examples/**`

### Excluded Paths

The following paths are excluded from analysis:
- `app/macos/**` - Incomplete macOS Xcode project
- `build/**` - Build artifacts
- `third_party/**` - Third-party dependencies
- Generated protobuf/gRPC files
- Node modules and frontend build output

## Why Not Custom Workflow?

We initially created a custom CodeQL workflow but removed it because:
1. GitHub's default setup is already enabled on this repository
2. Custom workflows conflict with default setup (cannot use both)
3. Default setup provides automatic updates and better integration

## Manual Configuration Required

⚠️ **Important**: To use the `codeql-config.yml` file, a repository admin must:

1. Go to **Settings** → **Code security and analysis**
2. Under **CodeQL analysis**, click **Set up** → **Advanced**
3. Select "Use configuration file" and specify: `.github/codeql/codeql-config.yml`

Alternatively, keep using the default setup and accept that it will try to build all languages (the Swift/Objective-C build will fail until the macOS app is complete, but JavaScript/TypeScript and Python will still be analyzed).

## Troubleshooting

### "CodeQL analyses from advanced configurations cannot be processed"

This error occurs when:
- Default CodeQL setup is enabled, AND
- A custom workflow or advanced config is detected

**Solution**: Choose one:
- **Option A**: Disable default setup and use custom workflow
- **Option B**: Remove custom workflow and use default setup (current choice)

### C++ Build Failures

C++ analysis may fail on non-macOS runners due to:
- Missing MLX framework (macOS only)
- Missing Metal development tools

This is expected - CodeQL will still analyze the source code structure even if the build fails.

## Future Improvements

Once the macOS app (`app/macos/MLXR.xcodeproj`) is complete:
- Remove `app/macos/**` from `paths-ignore`
- Enable Swift/Objective-C analysis
- Consider adding custom query suites for domain-specific checks
