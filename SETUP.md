# Setup and Configuration Notes

This document describes the setup changes and how the repository dependencies are managed.

## Overview

**Primary Purpose**: These changes were implemented to add support for **RTX 50 series GPUs (Blackwell architecture)**. The RTX 50 series requires PyTorch 2.7.0+ with CUDA 12.8+ for optimal performance. Additionally, Python 3.12 support was added, along with compatibility fixes for the CompVis/stable-diffusion repository.

## Repository Dependencies

The webui automatically clones the following repositories during setup (handled by `modules/launch_utils.py`):

### Core Repositories
- **stable-diffusion-webui-assets**: UI assets and resources
  - Repo: `https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git`
  - Location: `repositories/stable-diffusion-webui-assets/`
  - Commit: `6f7db241d2f8ba7457bac5ca9753331f0c266917`

- **stable-diffusion-stability-ai**: Core Stable Diffusion implementation
  - Default Repo: `https://github.com/CompVis/stable-diffusion.git` (fallback from Stability-AI)
  - Location: `repositories/stable-diffusion-stability-ai/`
  - Note: Automatically falls back to CompVis if Stability-AI repo is unavailable
  - Commit: Latest main (no specific commit required for CompVis)

- **generative-models**: Stable Diffusion XL (SDXL) implementation
  - Repo: `https://github.com/Stability-AI/generative-models.git`
  - Location: `repositories/generative-models/`
  - Commit: `45c443b316737a4ab6e40413d7794a7f5657c19f`

- **k-diffusion**: K-diffusion sampling methods
  - Repo: `https://github.com/crowsonkb/k-diffusion.git`
  - Location: `repositories/k-diffusion/`
  - Commit: `ab527a9a6d347f364e3d185ba6d714e22d80cb3c`

- **BLIP**: BLIP model for image captioning
  - Repo: `https://github.com/salesforce/BLIP.git`
  - Location: `repositories/BLIP/`
  - Commit: `48211a1594f1321b00f14c9f7a5b4813144b2fb9`

- **taming-transformers**: Taming Transformers for VQGAN
  - Repo: `https://github.com/CompVis/taming-transformers.git`
  - Location: `repositories/taming-transformers/`
  - Commit: `24268930bf1dce879235a7fddd0b2355b84d7ea6`

## Git Clone Handling

All repository cloning is handled automatically by `modules/launch_utils.py` during the `prepare_environment()` function. The `webui.sh` script does not need to handle git cloning directly - it simply runs `launch.py` which calls `prepare_environment()`.

### How It Works

1. When `launch.py` starts, it calls `prepare_environment()`
2. The function checks if each repository exists in `repositories/` directory
3. If a repository doesn't exist, it clones it using the `git_clone()` function
4. If a repository exists but is at a different commit, it updates to the correct commit
5. If the remote URL has changed, it updates the remote URL

### Fallback Mechanism

The stable-diffusion repository has a fallback mechanism:
- First attempts to clone from the configured repo (default: CompVis)
- If that fails and the repo was Stability-AI, automatically falls back to CompVis
- This ensures compatibility even if repositories become unavailable

## Python Version Support

### Supported Versions
- Python 3.7, 3.8, 3.9, 3.10, 3.11, **3.12** (added support)

### PyTorch Version Selection

**Note**: The PyTorch version selection logic was primarily added for RTX 50 series GPU support.

The system automatically selects the appropriate PyTorch version based on:
1. **RTX 50 series (Blackwell architecture) detection**: 
   - **Priority**: Highest priority check
   - If detected (compute capability 12.0), automatically uses PyTorch 2.7.0+ with CUDA 12.8
   - This is required for RTX 50 series GPUs to function properly
   - Detection is done via `nvidia-smi` querying compute capability
2. **Python version**: 
   - Python 3.12+: PyTorch 2.2.0+ with CUDA 12.1 (2.1.2 doesn't support Python 3.12)
   - Python 3.7-3.11: PyTorch 2.1.2 with CUDA 12.1 (default)

## RTX 50 Series GPU Support

### Automatic Detection and Configuration

The system automatically detects RTX 50 series GPUs (Blackwell architecture, compute capability 12.0) and configures the environment accordingly:

- **Detection Method**: Uses `nvidia-smi` to query GPU compute capability
- **Required PyTorch**: 2.7.0+ with CUDA 12.8
- **Required Torchvision**: 0.22.0+ with CUDA 12.8
- **Index URL**: `https://download.pytorch.org/whl/cu128`

If an RTX 50 series GPU is detected, the system will automatically install the correct PyTorch version during setup, overriding any default PyTorch installation.

## Compatibility Fixes

### CompVis Repository Compatibility

The codebase has been updated to work with the CompVis/stable-diffusion repository (which is now the default) instead of the deprecated Stability-AI repository. These compatibility fixes ensure the codebase works with both CompVis and Stability-AI repositories:

1. **Attribute checks**: Added `hasattr()` checks before accessing attributes that may not exist in CompVis:
   - `MemoryEfficientCrossAttention` 
   - `ATTENTION_MODES` in `BasicTransformerBlock`
   - `FrozenOpenCLIPEmbedder` in encoders

2. **Import handling**: Added `mute_sdxl_imports()` function to create dummy modules for SDXL imports that aren't actually needed

3. **Error handling**: Improved safetensors loading with fallback mechanisms for HeaderTooLarge errors

## Virtual Environments

### Main Virtual Environment
- Location: `venv/` (or configured via `venv_dir` in `webui-user.sh`)
- Managed automatically by `webui.sh`

### SDXL Virtual Environment (Optional)
- Location: `SDXL/`
- This is a separate Python virtual environment if you need to isolate SDXL dependencies
- **Note**: This directory is ignored by git (added to `.gitignore`)

## Environment Variables

You can customize repository URLs and commit hashes using environment variables:

```bash
export STABLE_DIFFUSION_REPO="https://github.com/CompVis/stable-diffusion.git"
export STABLE_DIFFUSION_COMMIT_HASH=""
export STABLE_DIFFUSION_XL_REPO="https://github.com/Stability-AI/generative-models.git"
export STABLE_DIFFUSION_XL_COMMIT_HASH="45c443b316737a4ab6e40413d7794a7f5657c19f"
export K_DIFFUSION_REPO="https://github.com/crowsonkb/k-diffusion.git"
export K_DIFFUSION_COMMIT_HASH="ab527a9a6d347f364e3d185ba6d714e22d80cb3c"
export BLIP_REPO="https://github.com/salesforce/BLIP.git"
export BLIP_COMMIT_HASH="48211a1594f1321b00f14c9f7a5b4813144b2fb9"
export TAMING_REPO="https://github.com/CompVis/taming-transformers.git"
export TAMING_COMMIT_HASH="24268930bf1dce879235a7fddd0b2355b84d7ea6"
```

## Troubleshooting

### Repository Clone Failures

If a repository fails to clone:
1. Check your internet connection
2. Verify the repository URL is accessible
3. For stable-diffusion, the system will automatically try the CompVis fallback
4. Check git is installed: `git --version`

### Python Version Issues

If you encounter Python version errors:
- Ensure you're using a supported Python version (3.7-3.12)
- For Python 3.12+, PyTorch 2.2.0+ will be automatically installed
- For RTX 50 series GPUs, PyTorch 2.7.0+ will be automatically installed

### RTX 50 Series GPU Issues

If you have an RTX 50 series GPU and encounter issues:
- Verify GPU detection: Run `nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits` - should show `12.0`
- Check PyTorch version: After installation, verify with `python -c "import torch; print(torch.__version__)"` - should be 2.7.0 or higher
- Verify CUDA version: Check with `python -c "import torch; print(torch.version.cuda)"` - should be 12.8 or compatible
- If detection fails, you can manually set: `export TORCH_COMMAND="pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128"`

### Safetensors Loading Errors

If you encounter HeaderTooLarge errors when loading models:
- The system will automatically try a fallback loading method
- Check that the model file is not corrupted (file size should be reasonable)
- Ensure you have enough RAM/VRAM to load the model
