# Packaging & Distribution Plan – MLXR (macOS, Apple Silicon)

This document defines how we build, sign, package, update, and distribute the MLXR: a macOS app bundle with a background daemon, CLI, SDKs, and kernels. Focus: Apple Silicon (M4+, supporting M2/M3); macOS 14+.

---

## 1) Artifacts & Targets

**Primary**

* `MLXR.app` – GUI bundle (React WebView + Swift/ObjC host).
* `mlxrunnerd` – background daemon (launchd agent, local REST/gRPC).
* `mlx` – CLI shim (manage models, call APIs, dev tools).
* `libmlxrunner_core.a` / `libmlxrunner_core.dylib` – engine core + MLX glue.
* `kernels.metallib` – compiled Metal kernels fat library (per‑variant slices).

**Secondary**

* Installer images: `.dmg` (drag‑and‑drop), `.pkg` (installer), `.zip` (portable).
* SDKs: `pip` wheel (`mlxrunner`), npm package (`@mlx/runner`), SwiftPM package.
* SBOM: SPDX JSON + CycloneDX for each release.

---

## 2) Build Matrix & Versioning

* **Arch**: arm64 only; build fails early on x86_64.
* **macOS min**: 14.0 (Sonoma) – using modern Metal & Sparkle.
* **SemVer**: `MAJOR.MINOR.PATCH` (e.g., `1.3.0`).
* **Channels**: `stable`, `beta` (feature flags), `nightly` (unsigned/notarized optional).
* **Metallib variants**: compiled by `scripts/build_metal.sh` → `kernels/{kernel}_{variant}.metallib`; combined into `kernels.metallib`.

---

## 3) Codesigning, Hardened Runtime, Notarization

**Identities**

* `Developer ID Application: <Team Name> (<TEAMID>)`
* `Developer ID Installer: <Team Name> (<TEAMID>)`

**Entitlements (app)**

* `com.apple.security.app-sandbox` (true)
* `com.apple.security.files.user-selected.read-write` (model dirs)
* `com.apple.security.network.client` (model pulls)
* `com.apple.security.network.server` (optional for localhost)
* `com.apple.security.cs.disable-library-validation` (false)
* `com.apple.security.automation.apple-events` (false)

**Entitlements (daemon)**

* sandboxed; `network.server` true; `network.client` true; no camera/mic.

**Process**

1. Sign all binaries (`codesign --options runtime --timestamp`).
2. Bundle Sparkle’s `SUUpdater` with EdDSA keys; sign app with Hardened Runtime.
3. Create notarization zip: `xcrun notarytool submit` → `xcrun stapler staple`.
4. Verify: `spctl --assess --type execute` and Gatekeeper trial.

**Privacy strings (Info.plist)**

* If no camera/mic/location used, omit; add NSAppTransportSecurity exceptions as needed for model mirrors.

---

## 4) App Bundle Layout

```
MLXR.app/
  Contents/
    MacOS/MLXR         # GUI host
    Frameworks/
    Resources/
      ui/                    # React assets
      kernels/kernels.metallib
      defaults/server.yaml
      licenses/
      sparkle/               # appcast, keys
    Info.plist
```

---

## 5) Daemon & Launchd

**Install location**: `/Library/MLXRunner/bin/mlxrunnerd` (system) or `~/Library/Application Support/MLXRunner/bin/mlxrunnerd` (user).

**Launch agent**: `~/Library/LaunchAgents/com.company.mlxrunnerd.plist`

```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.company.mlxrunnerd</string>
  <key>ProgramArguments</key>
  <array>
    <string>~/Library/Application Support/MLXRunner/bin/mlxrunnerd</string>
    <string>--config</string>
    <string>~/Library/Application Support/MLXRunner/server.yaml</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>~/Library/Logs/mlxrunnerd.out.log</string>
  <key>StandardErrorPath</key><string>~/Library/Logs/mlxrunnerd.err.log</string>
</dict></plist>
```

**Permissions**: daemon listens on UDS at `~/Library/Application Support/MLXRunner/run/mlxrunner.sock` with `0600`.

---

## 6) Installer Options

**DMG (drag‑and‑drop)**

* Pros: familiar UX, simple.
* Steps: mount DMG → drag `MLXR.app` to `/Applications` → first‑run helper installs daemon to user location, writes launch agent.

**PKG (recommended for managed/enterprise)**

* Installs app to `/Applications`, daemon to `/Library/MLXRunner`, creates launchd plist in `~/Library/LaunchAgents/`.
* Signed with `Developer ID Installer`; notarized.

**ZIP (portable)**

* For power users; no auto‑update.

---

## 7) Auto‑Updates (Sparkle)

* Appcast hosted over HTTPS.
* **Delta updates** enabled; full fallback builds per channel.
* EdDSA signing key baked into app; private key kept offline.
* Daemon updates: app orchestrates download, verifies signature and SHA‑256, swaps `mlxrunnerd` on next restart.

---

## 8) Model Registry, Caching & Paths

**Defaults**

* Models root: `~/Library/Application Support/MLXRunner/models/`
* Cache: `~/Library/Application Support/MLXRunner/cache/`
* Config: `~/Library/Application Support/MLXRunner/server.yaml`

**Integrity**

* Each model file accompanied by `.sha256` and optional `.sig` (Ed25519). App verifies on import/pull.

**Migration**

* On first run, migrate prior versions; maintain schema version in `registry.sqlite`.

---

## 9) CLI & SDK Distribution

**CLI**

* Install with app (symlink `/usr/local/bin/mlx` to app helper) or via Homebrew.
* Homebrew tap:

  * `brew tap company/mlxrunner`
  * `brew install mlxrunner` (cask installs app + symlink; formula installs CLI only).

**Python**

* `pip install mlxrunner` – wheels built via cibuildwheel (macOS arm64). Binary extension links to `libmlxrunner_core.dylib`.

**Node**

* `npm i @mlx/runner` – typed client for OpenAI/Ollama endpoints.

**SwiftPM**

* `https://github.com/company/mlxrunner-spm` – API client + local SDK wrappers.

---

## 10) CI/CD Pipeline (GitHub Actions)

**Jobs**

* `build_core`: CMake + Metal compile; cache DerivedData and metallib.
* `build_app`: Xcode build (release), codesign, archive.
* `package`: create DMG/PKG/ZIP; notarize; staple.
* `sdks`: publish PyPI (arm64 wheel), npm, SwiftPM tag.
* `release`: generate SBOM, checksums, GitHub Release assets, update Sparkle appcast.

**Key Steps**

* Use `macos-14` runners.
* Secrets: Apple API key for notarization, Sparkle private key (sign offline preferred), signing certs from keychain.
* Artifacts: `MLXR.dmg`, `MLXR.pkg`, `mlxrunnerd.zip`, `kernels.metallib`, SBOM.

---

## 11) Security & Trust

* **Hardened runtime** + sandbox for app and daemon.
* **Model verification**: SHA‑256 and optional signature check on import/pull.
* **Network**: default bind to UDS only; HTTP port disabled unless toggled.
* **Local auth**: capability token for CLI/UI; stored in Keychain, rotated on update.
* **Telemetry**: opt‑in; no PII; can be built without telemetry.
* **SBOM**: attach per release; use `syft` or `cyclonedx`.

---

## 12) Update & Migration Strategy

* Config migrations run on startup; schema version gate.
* Metallib compatibility: runtime verifies `kernels.metallib` version vs engine; if mismatch, soft‑fail and run MLX fallback.
* Rollback: keep previous app and daemon copies; Sparkle supports rollback via appcast.

---

## 13) First‑Run & Onboarding

* Welcome screen: choose models directory, toggle auto‑updates, opt‑in telemetry.
* Download starter model (optional); show disk usage.
* Validate GPU availability, compile a few pipeline states, warm caches.

---

## 14) Licensing & Legal

* License: Apache‑2.0 (example) + NOTICE file.
* Model licenses displayed before download; store acceptances with timestamp.
* Export controls: no remote telemetry of model names unless opted in.

---

## 15) Manual Verification Checklist

* Gatekeeper opens app without warnings (stapled).
* Daemon launches via launchd, creates UDS with `0600` perms.
* CLI reachable from `$PATH`.
* OpenAI & Ollama endpoints respond locally.
* Auto‑update from `1.0.0` → `1.0.1` delta works; rollback path verified.
* Model import verifies checksum/signature; corrupt file rejected.

---

## 16) Release Playbook

1. Bump version & changelog.
2. Tag and push; CI builds, signs, notarizes.
3. QA on clean macOS VM (Sonoma & Sequoia if available).
4. Publish GitHub Release with DMG/PKG/ZIP, checksums, SBOM.
5. Update Sparkle appcast (stable/beta); verify delta signatures.
6. Publish Homebrew cask PR + PyPI + npm + SwiftPM tag.
7. Announce and update docs site.

---

## 17) Uninstall & Cleanup

* Remove `MLXR.app`.
* `launchctl unload ~/Library/LaunchAgents/com.company.mlxrunnerd.plist && rm` plist.
* Delete support dir: `~/Library/Application Support/MLXRunner/` (prompt user if models present).

---

## 18) Open Questions

* Enterprise MDM distribution: provide `.pkg` with configuration profile? (future)
* Optional system service vs user agent by default?
