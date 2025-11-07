# Test Models

This directory contains small language models downloaded from HuggingFace for CI testing and development.

## Model Storage

Models are stored using **Git LFS** (Large File Storage) to efficiently version large binary files without bloating the repository.

### Tracked Extensions

The following file types are automatically tracked by Git LFS (see `.gitattributes`):

- `*.safetensors` - Model weights in safetensors format
- `*.gguf` - Quantized GGUF model files
- `*.bin` - PyTorch model weights

Small files (`.json`, `.txt`, `.md`) are stored in regular git.

## Available Models

### TinyLlama 1.1B Chat (Safetensors)

- **Directory**: `test_models/tinyllama/`
- **Repository**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Format**: Safetensors (FP16)
- **Size**: ~4.4 GB
- **Use Case**: Basic inference testing, model loading validation

### TinyLlama 1.1B Q4_K_M (GGUF)

- **Directory**: `test_models/tinyllama-gguf/`
- **Repository**: [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- **Format**: GGUF Q4_K_M quantized
- **Size**: ~669 MB
- **Use Case**: Quantized model testing, GGUF parser validation

## Downloading Models

### Manual Download

Use the download script to fetch models locally:

```bash
# Download all configured models
python scripts/download_test_models.py

# Download specific model
python scripts/download_test_models.py --model tinyllama

# Force re-download
python scripts/download_test_models.py --force

# List available models
python scripts/download_test_models.py --list
```

### Automated Download (CI)

Models are automatically downloaded via the **Download Test Models** workflow (`.github/workflows/download-models.yml`):

- **Trigger**: Manual dispatch or weekly schedule (Sundays at 2 AM UTC)
- **Artifacts**: Models are uploaded as workflow artifacts (30-day retention)
- **Git LFS**: Models are committed to the repository via Git LFS

#### Using Workflow Artifacts

Other workflows can download model artifacts instead of re-downloading from HuggingFace:

```yaml
- name: Download test models artifact
  uses: actions/download-artifact@v4
  with:
    name: test-models-${{ env.MODEL_ARTIFACT_RUN_ID }}
    path: test_models/
```

## Model Manifests

Each downloaded model includes a `manifest.json` file with metadata:

```json
{
  "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "description": "TinyLlama 1.1B - Small model for basic testing",
  "size_mb": 4400,
  "files": {
    "model.safetensors": "sha256:abc123...",
    "config.json": "sha256:def456...",
    "tokenizer.model": "sha256:789xyz..."
  }
}
```

Manifests enable:
- File integrity verification via SHA256 hashes
- Cache validation (skip re-download if unchanged)
- Model metadata tracking

## Git LFS Setup

If you don't have Git LFS installed:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Initialize LFS in the repository
git lfs install
```

## Pulling Models

When cloning the repository or switching branches:

```bash
# Pull LFS files
git lfs pull

# Verify LFS status
git lfs ls-files
```

## Adding New Models

To add a new test model:

1. Edit `scripts/download_test_models.py` and add to `TEST_MODELS` dict:

```python
"my-model": {
    "repo_id": "author/model-name",
    "allow_patterns": ["*.json", "*.safetensors", "tokenizer.model"],
    "ignore_patterns": ["*.bin"],
    "description": "Description for documentation",
    "size_mb": 1234,
}
```

2. Run the download script:

```bash
python scripts/download_test_models.py --model my-model
```

3. Commit the changes:

```bash
git add test_models/my-model/
git commit -m "Add my-model test model"
git push
```

Git LFS will automatically handle the large files.

## Storage Considerations

- **Git LFS Quota**: GitHub provides 1 GB storage + 1 GB bandwidth/month for free
- **Current Usage**: ~5 GB for all test models (TinyLlama safetensors + GGUF)
- **Bandwidth**: LFS files count against bandwidth when pulled
- **Artifacts**: Workflow artifacts stored separately (count against Actions storage)

### Recommendations

- Use workflow artifacts in CI (doesn't count against LFS bandwidth)
- Only pull LFS files when needed: `GIT_LFS_SKIP_SMUDGE=1 git clone`
- Consider model size when adding new test models
- Prefer quantized models (GGUF Q4_K_M) when possible

## Troubleshooting

### Models not downloading in CI

Check that the workflow has `lfs: true` in checkout:

```yaml
- uses: actions/checkout@v4
  with:
    lfs: true
```

### Large repository clone times

Skip LFS download initially:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repo>
cd <repo>
git lfs pull  # Pull LFS files when needed
```

### LFS quota exceeded

Use workflow artifacts instead of pulling from LFS:

```yaml
- uses: actions/download-artifact@v4
  with:
    name: test-models-latest
```
