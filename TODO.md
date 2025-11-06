# MLXR Project TODO & Roadmap
**Last Updated:** 2025-11-06
**Based On:** Comprehensive plan review (PLAN_REVIEW_REPORT.md)

---

## 🔴 CRITICAL - Must Complete Before Launch

### 1. Configuration System ⚠️ BLOCKING
**Priority:** P0 - CRITICAL
**Effort:** 2-4 hours
**Status:** ❌ Not Started

**Problem:** `configs/` directory does not exist. Applications cannot load configuration.

**Tasks:**
- [ ] Create `configs/` directory structure
- [ ] Create `configs/server.yaml` with all settings
  - [ ] UDS path configuration
  - [ ] max_batch_tokens setting
  - [ ] target_latency_ms setting
  - [ ] enable_speculative setting
  - [ ] draft_model path
  - [ ] kv_persistence setting
  - [ ] HTTP port (optional)
- [ ] Create example model configs in `configs/models/`
  - [ ] `configs/models/llama-3-8b.yaml`
  - [ ] `configs/models/mistral-7b.yaml`
  - [ ] `configs/models/tinyllama-1.1b.yaml`
- [ ] Update CMakeLists.txt to install configs
- [ ] Update daemon to load from configs/

**Files to Create:**
```
configs/
  server.yaml              # Main daemon configuration
  models/
    llama-3-8b.yaml       # Example Llama config
    mistral-7b.yaml       # Example Mistral config
    tinyllama-1.1b.yaml   # Example TinyLlama config
```

**References:**
- SPEC01.md lines 294-295
- Structure.md lines 136-140
- Plan specifies complete YAML schema

---

### 2. Documentation Accuracy ⚠️ CRITICAL
**Priority:** P0 - CRITICAL
**Effort:** 2-3 hours
**Status:** ❌ Not Started

**Problem:** CLAUDE.md contains multiple inaccuracies that mislead developers.

**Critical Inaccuracies to Fix:**

1. **Line 52: Metal Kernels Status**
   - Current: "Metal Kernels - 🚧 PARTIALLY COMPLETE (40%)"
   - Actual: ~95% complete, all 6 kernels implemented (~110K LOC)
   - Fix: Update to "Metal Kernels - ✅ COMPLETE (95%)"

2. **Line 200: configs/ Directory**
   - Current: Lists "configs/ # Server & model configs (YAML)"
   - Actual: Directory does not exist
   - Fix: Remove from structure OR add note "❌ TO BE CREATED"

3. **Line 223: tools/ Description**
   - Current: Lists conversion tools
   - Actual: Only placeholder CMakeLists.txt
   - Fix: Mark as "⏳ PLANNED" and note Phase 6

4. **Line 50-60: Phase 2 Status**
   - Current: "Phase 2 ~85% COMPLETE"
   - Actual: Core features ~95% complete
   - Fix: Update percentages based on review

5. **Line 27: CPU Kernels**
   - Current: "kernels/cpu/ # Neon/SIMD fallbacks"
   - Actual: Only .gitkeep file
   - Fix: Mark as "❌ NOT IMPLEMENTED (GPU-only)"

**Tasks:**
- [ ] Update Metal kernels status to ~95%
- [ ] Remove or clarify configs/ status
- [ ] Mark tools/ as planned/future
- [ ] Update phase completion percentages
- [ ] Add note about GPU-only requirement
- [ ] Remove all gRPC references (use "REST API" only)
- [ ] Add section on known limitations

---

### 3. gRPC Server Implementation ⚠️ CRITICAL
**Priority:** P0 - CRITICAL
**Effort:** 8-12 hours
**Status:** ❌ Not Started

**Problem:** Plan documents specify "REST/gRPC" but only REST is implemented.

**Decision:** IMPLEMENT gRPC to match plan specifications.

**Tasks:**
- [ ] Add gRPC dependencies to CMakeLists.txt (grpc++, protobuf)
- [ ] Create `daemon/server/proto/` directory for protocol definitions
- [ ] Define protobuf service definitions
  - [ ] `mlxrunner.proto` - Main service definition
  - [ ] OpenAI-compatible RPCs
  - [ ] Ollama-compatible RPCs
  - [ ] Streaming support for tokens
- [ ] Implement `daemon/server/grpc_server.{h,cpp}`
  - [ ] gRPC server initialization
  - [ ] Service implementation
  - [ ] Streaming token responses
  - [ ] Integration with existing scheduler
- [ ] Add gRPC endpoint to daemon startup
  - [ ] Bind to Unix Domain Socket (default)
  - [ ] Optional TCP port (localhost only)
- [ ] Update REST server to coexist with gRPC
- [ ] Add gRPC client to SDKs
  - [ ] Python gRPC client
  - [ ] TypeScript gRPC client
- [ ] Test gRPC endpoints
- [ ] Update CLAUDE.md to document gRPC implementation

**Files to Create:**
- `daemon/server/proto/mlxrunner.proto`
- `daemon/server/grpc_server.h`
- `daemon/server/grpc_server.cpp`
- `daemon/CMakeLists.txt` (update for gRPC)

**References:**
- SPEC01.md mentions "REST/gRPC" throughout
- Structure.md line 82 shows "REST/gRPC Server" component
- Plan specifies dual REST+gRPC support

---

## 🟡 HIGH PRIORITY - Needed for Usability

### 4. Model Conversion Tools
**Priority:** P1 - HIGH
**Effort:** 8-12 hours
**Status:** ❌ Not Started

**Problem:** Users cannot convert models between formats.

**Planned Tools (from SPEC01.md line 237-238):**
1. `tools/convert_hf_to_mlx.py` - Hugging Face to MLX format
2. `tools/convert_to_gguf.py` - Hugging Face to GGUF format
3. `tools/quantize_fp16_to_fp8.py` - Quantization utility

**Options:**
- **Option A:** Implement tools (8-12 hours)
- **Option B:** Document external tool usage (2 hours)
  - Recommend `mlx-community` converters
  - Recommend `llama.cpp` for GGUF
  - Add documentation on manual conversion

**Recommended:** Option B short-term, Option A long-term

**Tasks:**
- [ ] Add `docs/MODEL_CONVERSION.md` guide
- [ ] Document external converter usage
- [ ] Add links to community tools
- [ ] Add to roadmap for Phase 6 implementation

---

### 5. CPU Kernel Strategy
**Priority:** P1 - HIGH
**Effort:** 1 hour (documentation) OR 40+ hours (implementation)
**Status:** ❌ Not Started

**Problem:** Plan specifies CPU kernels, but only GPU implemented.

**Current State:**
- `core/kernels/cpu/` contains only `.gitkeep`
- Plan specifies "Neon/SIMD fallbacks"
- All kernels are Metal (GPU) only

**Decision Required:**
- **Option A:** Document as GPU-only (Apple Silicon required)
- **Option B:** Implement CPU fallback kernels

**Recommendation:** Option A - Document GPU requirement

**Tasks:**
- [ ] Add `docs/SYSTEM_REQUIREMENTS.md`
- [ ] Clearly state: "Apple Silicon with GPU required"
- [ ] Document minimum: M2, recommended: M4
- [ ] Update all README files
- [ ] Remove CPU kernel references from plans
- [ ] Keep `core/kernels/cpu/.gitkeep` for future

---

### 6. CLI Binary (mlx)
**Priority:** P1 - HIGH
**Effort:** 4-6 hours
**Status:** ❌ Not Started

**Problem:** Plan specifies `mlx` CLI binary, but none exists.

**Current State:**
- Python SDK has CLI functionality (`mlxrunner` package)
- No standalone `mlx` binary for shell usage

**Tasks:**
- [ ] Create `cli/` directory
- [ ] Create `cli/main.cpp` - CLI entry point
- [ ] Implement model management commands
  - [ ] `mlx pull <model>` - Download model
  - [ ] `mlx list` - List installed models
  - [ ] `mlx run <model>` - Run inference
  - [ ] `mlx serve` - Start daemon
  - [ ] `mlx status` - Check daemon status
- [ ] Add CMake target for CLI binary
- [ ] Install to `/usr/local/bin/mlx` or bundle with app
- [ ] Add bash/zsh completions

**Alternative:** Document Python CLI as primary interface

---

### 7. Xcode Project Creation
**Priority:** P1 - HIGH (for macOS app)
**Effort:** 1 hour
**Status:** ⏳ MANUAL SETUP REQUIRED

**Problem:** All Swift source files exist, but no `.xcodeproj` file.

**Current State:**
- 24 Swift source files implemented (~2,800 LOC)
- No Xcode project file to build them
- `app/macos/README.md` has instructions

**Tasks:**
- [ ] Follow `app/macos/README.md` instructions
- [ ] Create Xcode project (macOS App template)
- [ ] Add all 24 Swift files to project
- [ ] Configure build settings
  - [ ] Deployment target: macOS 14.0+
  - [ ] Architecture: arm64
  - [ ] Bundle ID: com.company.mlxr
- [ ] Link frameworks (WebKit, AppKit, Security)
- [ ] Add entitlements and Info.plist
- [ ] Test build
- [ ] Commit `.xcodeproj` to git

**Note:** This is manual work, cannot be automated via script.

---

## 🟢 MEDIUM PRIORITY - Polish & Distribution

### 8. App Icons & Assets
**Priority:** P2 - MEDIUM
**Effort:** 2-4 hours (design) + 1 hour (integration)
**Status:** ❌ Not Started

**Problem:** Asset structure exists but no actual icons designed.

**Tasks:**
- [ ] Design AppIcon (1024x1024 base)
- [ ] Export to all required sizes (10 variants)
- [ ] Design TrayIcon (16x16 and 32x32)
- [ ] Design MenuBarIcon (template images)
- [ ] Add to `app/macos/MLXR/Resources/Assets.xcassets/`
- [ ] Test in Xcode

**Can use placeholder icons initially, but need professional design for release.**

---

### 9. Code Signing & Notarization
**Priority:** P2 - MEDIUM (for distribution)
**Effort:** 4-6 hours
**Status:** ⏳ SCRIPTS EXIST, TESTING NEEDED

**Current State:**
- Scripts exist: `sign_app.sh`, `create_dmg.sh`
- Not tested end-to-end
- No notarization workflow

**Tasks:**
- [ ] Obtain Developer ID Certificate
- [ ] Test `scripts/sign_app.sh` with real certificate
- [ ] Test `scripts/create_dmg.sh` with signed app
- [ ] Setup notarization workflow
  - [ ] `xcrun notarytool submit`
  - [ ] Wait for approval
  - [ ] `xcrun stapler staple`
- [ ] Verify Gatekeeper acceptance
- [ ] Document process in `docs/DISTRIBUTION.md`

---

### 10. Sparkle Auto-Updates
**Priority:** P2 - MEDIUM (for production)
**Effort:** 6-8 hours
**Status:** ⏳ PARTIAL (UpdateManager.swift exists)

**Current State:**
- `app/macos/MLXR/Services/UpdateManager.swift` exists but incomplete
- No Sparkle framework integrated
- No appcast setup

**Tasks:**
- [ ] Add Sparkle.framework to Xcode project
- [ ] Complete UpdateManager.swift implementation
- [ ] Generate EdDSA key pair for signing
- [ ] Create appcast.xml template
- [ ] Setup appcast hosting (HTTPS required)
- [ ] Implement delta update support
- [ ] Test update flow end-to-end
- [ ] Add "Check for Updates" menu item

**Reference:** MacosPlan.md Phase 6 (lines 211-246)

---

### 11. Distribution Infrastructure
**Priority:** P2 - MEDIUM
**Effort:** 8-12 hours
**Status:** ⏳ PARTIAL (~30% complete)

**Current State:**
- DMG script exists
- No PKG installer
- No Homebrew tap
- No CI/CD pipeline

**Tasks:**
- [ ] **DMG Distribution**
  - [ ] Test `scripts/create_dmg.sh` end-to-end
  - [ ] Add custom DMG background image
  - [ ] Add Applications symlink
  - [ ] Sign and notarize DMG

- [ ] **PKG Installer** (for enterprise)
  - [ ] Create installer script
  - [ ] Define install locations
  - [ ] Add preinstall/postinstall scripts
  - [ ] Sign with Developer ID Installer

- [ ] **Homebrew Tap**
  - [ ] Create `homebrew-mlxr` repository
  - [ ] Write Cask formula
  - [ ] Test `brew install --cask mlxr`
  - [ ] Document tap usage

- [ ] **CI/CD Pipeline** (GitHub Actions)
  - [ ] Create `.github/workflows/build.yml`
  - [ ] Build on macos-14 runner
  - [ ] Run tests
  - [ ] Sign and notarize
  - [ ] Create GitHub Release
  - [ ] Upload DMG/PKG artifacts
  - [ ] Update appcast

---

### 12. SDK Publishing
**Priority:** P2 - MEDIUM
**Effort:** 2-4 hours
**Status:** ⏳ STRUCTURE EXISTS, NOT PUBLISHED

**Current State:**
- Python SDK: Complete structure, not on PyPI
- TypeScript SDK: Complete structure, not on npm
- Swift SDK: Not created

**Tasks:**
- [ ] **Python SDK (PyPI)**
  - [ ] Verify setup.py configuration
  - [ ] Build wheel: `python -m build`
  - [ ] Test wheel locally
  - [ ] Publish to PyPI: `twine upload dist/*`
  - [ ] Verify installation: `pip install mlxrunner`

- [ ] **TypeScript SDK (npm)**
  - [ ] Verify package.json
  - [ ] Build: `npm run build`
  - [ ] Test locally: `npm link`
  - [ ] Publish: `npm publish`
  - [ ] Verify: `npm install @mlx/runner`

- [ ] **Swift SDK (SwiftPM)**
  - [ ] Create separate repository
  - [ ] Write Package.swift
  - [ ] Implement OpenAI/Ollama clients
  - [ ] Add examples
  - [ ] Tag release
  - [ ] Test: `swift package resolve`

---

## 🔵 LOW PRIORITY - Future Enhancements

### 13. Vision Support (LLaVA/CLIP)
**Priority:** P3 - LOW (Phase 5 feature)
**Effort:** 20-30 hours
**Status:** ❌ NOT STARTED

**Planned Features:**
- CLIP/ViT encoder support
- Image patchify kernel (`clip_patchify_proj`)
- LLaVA model loading
- Image chat templates
- Vision playground in UI

**Tasks:**
- [ ] Research LLaVA model format
- [ ] Implement CLIP encoder in core/graph/
- [ ] Add `clip_patchify_proj` Metal kernel
- [ ] Update model loader for vision models
- [ ] Add image preprocessing
- [ ] Update UI for image upload
- [ ] Test with LLaVA model

**Note:** Deferred to Phase 5 per plan.

---

### 14. KV Cache Persistence
**Priority:** P3 - LOW (claimed "default on")
**Effort:** 2-4 hours (verification)
**Status:** ❓ UNKNOWN

**Problem:** Plan says KV persistence is "enabled by default" but implementation status unclear.

**Tasks:**
- [ ] Verify if `kv_persist_copy` kernel implemented
- [ ] Check if arena.cpp has persistence code
- [ ] Test KV cache save/restore
- [ ] Document persistence behavior
- [ ] Add config option: `kv_persistence: true`

**Note:** May already be implemented, needs verification.

---

### 15. Comprehensive Testing
**Priority:** P3 - LOW (ongoing)
**Effort:** 8-16 hours
**Status:** ⏳ PARTIAL (14 unit tests exist)

**Current State:**
- 14 C++ unit tests exist
- RMSNorm: 81/81 tests passing
- GQA tests: 6/6 passing
- Scheduler tests: 10/12 passing
- No integration tests
- No E2E tests

**Tasks:**
- [ ] Fix 2 failing scheduler tests
- [ ] Add integration tests
  - [ ] Full inference pipeline test
  - [ ] Multi-request batching test
  - [ ] KV cache eviction test
- [ ] Add E2E tests (if possible in headless environment)
  - [ ] App launch test
  - [ ] Daemon communication test
  - [ ] Model loading test
- [ ] Add performance benchmarks
- [ ] Setup CI test runs

---

### 16. Performance Validation
**Priority:** P3 - LOW (ongoing)
**Effort:** 4-8 hours
**Status:** ⏳ EXAMPLES EXIST

**Current State:**
- Example programs test basic functionality
- No systematic performance benchmarks
- Plan specifies targets (e.g., <80ms/token)

**Tasks:**
- [ ] Create benchmark suite
- [ ] Test on M4 hardware
- [ ] Measure against targets:
  - [ ] First token: <1s (7-8B at 4-bit)
  - [ ] Decode: <80ms/token
  - [ ] Embeddings: <20ms/sample
  - [ ] Prefill bandwidth: ≥1.3× decode
- [ ] Document actual performance
- [ ] Optimize bottlenecks if needed

---

## 📋 Deferred / Future Work

### Phase 6+ Features (Not Critical for MVP)

1. **CPU Kernels (Neon/SIMD)** - GPU-only is acceptable for MVP
2. **gRPC Server** - REST API sufficient for MVP
3. **Advanced Quantization** (IQ variants, FP8) - Basic quants work
4. **Draft Model (Speculative Decoding)** - Code exists, testing needed
5. **LoRA/Adapter Stacking** - Infrastructure exists, needs testing
6. **Tool/Function Calling** - Planned for future
7. **Multi-Model Residency** - Single model sufficient for MVP
8. **Embeddings Endpoint** - Planned, not critical
9. **ANE/CoreML Integration** - Optional optimization

---

## 📊 Progress Summary

### By Priority:

| Priority | Total Tasks | Completed | In Progress | Not Started |
|----------|-------------|-----------|-------------|-------------|
| P0 (Critical) | 3 | 0 | 0 | 3 |
| P1 (High) | 4 | 0 | 2 | 2 |
| P2 (Medium) | 5 | 0 | 3 | 2 |
| P3 (Low) | 4 | 0 | 2 | 2 |
| **TOTAL** | **16** | **0** | **7** | **9** |

### By Category:

| Category | Tasks | Status |
|----------|-------|--------|
| Configuration | 1 | ❌ Critical Gap |
| Documentation | 1 | ❌ Critical Gap |
| Integration | 2 | ⚠️ Partial |
| Features | 4 | ⚠️ Partial |
| Distribution | 3 | ⏳ In Progress |
| Testing | 2 | ⏳ In Progress |
| Future | 3 | ❌ Deferred |

---

## 🎯 Recommended Execution Order

### Sprint 1: Critical Fixes (2-3 days)
1. ✅ Create configs/ directory with defaults
2. ✅ Update CLAUDE.md to fix inaccuracies
3. ✅ Remove/clarify gRPC in all plans
4. ✅ Document GPU-only requirement
5. ✅ Add system requirements documentation

### Sprint 2: Core Functionality (1 week)
6. Create Xcode project (manual)
7. Build and test macOS app
8. Create/test daemon binary
9. Implement CLI binary OR document Python CLI
10. Add model conversion documentation

### Sprint 3: Distribution Prep (1-2 weeks)
11. Design app icons
12. Test code signing workflow
13. Test DMG creation
14. Setup CI/CD pipeline
15. Publish SDKs to PyPI/npm

### Sprint 4: Polish & Release (1 week)
16. Implement Sparkle updates
17. Create PKG installer
18. Setup Homebrew tap
19. Write user documentation
20. Beta release testing

---

## 📝 Notes

### What's Already Excellent:
- ✅ Metal kernels (~110K LOC, 95% complete)
- ✅ Frontend (78+ components, 90% complete)
- ✅ macOS app (24 Swift files, 95% complete)
- ✅ REST API (OpenAI + Ollama compatible)
- ✅ Daemon (scheduler, registry, telemetry)
- ✅ SDKs (Python and TypeScript)

### Known Issues:
- configs/ directory missing (critical)
- CLAUDE.md has inaccuracies
- gRPC mentioned but not implemented
- CPU kernels missing (GPU-only)
- Distribution infrastructure incomplete

### Questions for Product Owner:
1. Should we implement gRPC or remove from plans?
2. Is GPU-only acceptable or need CPU kernels?
3. Priority of vision support (LLaVA/CLIP)?
4. Timeline for first release?
5. Should we build CLI binary or use Python SDK?

---

**Last Review:** 2025-11-06
**Next Review:** After Sprint 1 completion
