# GitHub Workflows for MLXR

This directory contains GitHub Actions workflows for continuous integration, testing, and release automation.

## Workflows

### 1. CI - Build and Test (`ci.yml`)

**Triggers:**
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual dispatch

**Jobs:**
1. **metal-kernels**: Compiles all Metal shaders (.metal → .metallib)
2. **cpp-build**: Builds C++ core libraries and binaries
3. **cpp-tests**: Runs C++ unit tests (Google Test)
4. **python-tests**: Runs Python tests (pytest)
5. **code-quality**: Linting and formatting checks (Black, Ruff, MyPy, clang-format)

**Artifacts:**
- Metal kernel libraries (*.metallib)
- C++ build artifacts (libraries and binaries)
- Test results (XML format)

**Requirements:**
- macOS 14 (Sonoma) runner with Apple Silicon
- Xcode with Metal compiler
- MLX framework via Homebrew
- SentencePiece via Homebrew (for tokenization)
- nlohmann-json via Homebrew (for JSON parsing)
- cpp-httplib via Homebrew (for HTTP server)

**Example:** Every push to a branch automatically:
- Compiles Metal kernels
- Builds C++ code
- Runs all tests
- Checks code quality

### 2. Release - Build DMG (`release.yml`)

**Triggers:**
- Push of version tags (e.g., `v1.0.0`)
- Manual dispatch with version input

**Jobs:**
1. **build-app**: Complete macOS application build
   - Compiles Metal kernels
   - Builds C++ core and daemon
   - Builds React UI frontend
   - Builds Xcode project
   - Bundles all components
   - Signs application (ad-hoc for now)
   - Creates DMG installer

2. **create-release**: Creates GitHub release (on tags)
   - Uploads DMG to release
   - Generates release notes

**Artifacts:**
- DMG installer (MLXR-{version}-macOS-arm64.dmg)
- App bundle (MLXR.app)

**Requirements:**
- macOS 14 runner
- Node.js for UI build
- SentencePiece for tokenization
- XcodeGen (optional, for project generation)

**Usage:**
```bash
# Create a release
git tag v1.0.0
git push origin v1.0.0

# Or trigger manually from Actions tab
```

### 3. Daemon Integration Tests (`daemon-test.yml`)

**Triggers:**
- Push to `main` or `develop` affecting daemon code
- Pull requests affecting daemon code
- Manual dispatch

**Jobs:**
1. **build-daemon**: Builds daemon binary and dependencies
2. **daemon-unit-tests**: Runs daemon-specific unit tests
3. **daemon-integration-test**: Integration testing
   - Starts test daemon
   - Tests REST API endpoints
   - Tests health checks
   - Tests Ollama API compatibility
4. **daemon-stress-test**: Load testing with ApacheBench

**Test Coverage:**
- Scheduler tests
- Worker thread tests
- REST server tests
- Metrics collection tests
- Model registry tests

**Requirements:**
- macOS 14 runner
- MLX, SentencePiece, nlohmann-json, and cpp-httplib via Homebrew
- ApacheBench for stress testing

**Example Tests:**
```bash
# Health check
curl http://127.0.0.1:11434/health

# Models endpoint
curl http://127.0.0.1:11434/v1/models

# Ollama API
curl http://127.0.0.1:11434/api/tags
```

## Workflow Dependencies

```
ci.yml
├── metal-kernels (compiles Metal shaders)
│   └── cpp-build (uses Metal artifacts)
│       ├── cpp-tests (uses build artifacts)
│       └── python-tests (uses build artifacts)
└── code-quality (independent)

release.yml
└── build-app (full pipeline)
    └── create-release (creates GitHub release)

daemon-test.yml
└── build-daemon
    ├── daemon-unit-tests
    ├── daemon-integration-test
    └── daemon-stress-test
```

## Local Testing

You can replicate workflow steps locally:

```bash
# Metal compilation
./scripts/build_metal.sh

# C++ build
make build

# Run tests
make test-cpp
make test

# Build daemon
make test_daemon

# Start daemon for testing
./build/cmake/bin/test_daemon &
curl http://127.0.0.1:11434/health

# Build DMG
./scripts/create_dmg.sh
```

## Artifacts and Retention

| Artifact | Retention | Size (approx) |
|----------|-----------|---------------|
| Metal kernels | 7 days | 10-50 MB |
| C++ build artifacts | 7 days | 100-200 MB |
| Test results | 30 days | < 1 MB |
| DMG installer | 90 days | 500-1000 MB |
| App bundle | 30 days | 400-800 MB |

## Monitoring and Debugging

### Check Workflow Status

1. Go to **Actions** tab in GitHub repository
2. Select workflow from left sidebar
3. Click on specific run to see details

### View Logs

- Click on any job to expand steps
- Each step shows real-time or historical logs
- Failed steps are highlighted in red

### Download Artifacts

1. Go to workflow run page
2. Scroll to **Artifacts** section at bottom
3. Click artifact name to download

### Common Issues

**Metal compilation fails:**
- Check Xcode version: `xcodebuild -version`
- Verify Metal compiler: `xcrun metal --version`
- Review shader syntax in `core/kernels/metal/`

**C++ build fails:**
- Check CMake configuration in `CMakeLists.txt`
- Verify dependencies: `brew list`
- Check MLX installation: `python -c "import mlx"`
- Ensure SentencePiece is installed: `brew list sentencepiece`
- Ensure nlohmann-json is installed: `brew list nlohmann-json`
- Ensure cpp-httplib is installed: `brew list cpp-httplib`

**Daemon fails to start:**
- Check for port conflicts: `lsof -i :11434`
- Review daemon logs in workflow output
- Verify binary architecture: `file build/cmake/bin/test_daemon`

**DMG creation fails:**
- Check app bundle structure
- Verify code signing
- Check disk space on runner

## Performance Considerations

### Runner Performance

- **macOS runners**: GitHub provides M1 runners (macos-14)
- **Parallel jobs**: Jobs run concurrently when dependencies allow
- **Caching**: Consider adding cache for:
  - Homebrew packages
  - Conda environments
  - Node modules
  - CMake build cache

### Optimization Tips

1. **Use artifacts**: Pass build outputs between jobs instead of rebuilding
2. **Conditional jobs**: Use `if:` conditions to skip unnecessary jobs
3. **Matrix builds**: Test multiple configurations in parallel (future)
4. **Caching**: Add caching for dependencies (future enhancement)

## Security

### Code Signing

Currently using **ad-hoc signing** (`-`) for development builds.

For production releases:
1. Set up Apple Developer account
2. Generate certificates
3. Store as GitHub secrets
4. Update `SIGNING_IDENTITY` in release.yml
5. Enable notarization

### Secrets Management

Store sensitive data in GitHub Secrets:
- `APPLE_DEVELOPER_ID`: Developer ID certificate
- `NOTARIZATION_PASSWORD`: App-specific password for notarization
- `KEYCHAIN_PASSWORD`: Password for temporary keychain

### Permissions

Workflows have minimal permissions by default. The `create-release` job requires `contents: write` to create releases.

## Future Enhancements

### Planned Additions

1. **Caching Strategy**
   ```yaml
   - uses: actions/cache@v4
     with:
       path: ~/Library/Caches/Homebrew
       key: homebrew-${{ runner.os }}
   ```

2. **Matrix Testing**
   - Multiple macOS versions
   - Different model architectures
   - Various Python versions

3. **Performance Benchmarking**
   - Automated benchmark runs
   - Performance regression detection
   - Comparison with previous releases

4. **Notarization Pipeline**
   - Automated Apple notarization
   - Stapling for offline verification
   - Certificate management

5. **Docker Support** (future, if ARM Docker matures)
   - Containerized builds
   - Cross-platform testing

6. **Documentation Generation**
   - Auto-generate API docs
   - Deploy to GitHub Pages

## Contributing

When adding new workflows:

1. Test locally first using `act` or similar tools
2. Use descriptive job and step names
3. Add comments for complex steps
4. Update this README
5. Keep workflows focused (one concern per workflow)
6. Use artifacts to pass data between jobs
7. Add appropriate error handling

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [macOS Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners)
- [Artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
