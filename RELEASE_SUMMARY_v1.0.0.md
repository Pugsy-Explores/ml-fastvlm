# FastVLM v1.0.0 - Release Summary

## Tag Created
- **Tag**: `v1.0.0`
- **Commit**: `592b4ad` (first commit)
- **Date**: January 2025

## Release Highlights

### Core Features
✅ **FastViTHD Vision Encoder** - Novel hybrid architecture for efficient vision encoding  
✅ **3 Model Variants** - 0.5B, 1.5B, 7B parameter models with Stage 2 & 3 checkpoints  
✅ **PyTorch Inference** - Full CLI and programmatic inference support  
✅ **Apple Silicon Export** - MLX conversion tools with quantization (4/8-bit, FP16)  
✅ **iOS/macOS App** - Native Swift app with camera integration and real-time inference  
✅ **Serving Infrastructure** - Gradio web server, model workers, SGLang support  
✅ **Training Pipeline** - Complete LLaVA-based training codebase

### Performance
- 85x faster TTFT vs LLaVA-OneVision-0.5B
- 3.4x smaller vision encoder
- 7.9x faster TTFT for 7B vs Cambrian-1-8B

### Documentation
- Comprehensive README with setup instructions
- Model export guide for Apple Silicon
- iOS app setup and customization guide
- Research paper (CVPR 2025)

## Next Steps

To publish this release:

```bash
# Push tag to remote
git push origin v1.0.0

# Create GitHub release (via web UI or gh CLI)
gh release create v1.0.0 \
  --title "FastVLM v1.0.0" \
  --notes-file RELEASE_NOTES_v1.0.0.md
```

## Files Created
- `RELEASE_NOTES_v1.0.0.md` - Comprehensive release notes
- `RELEASE_SUMMARY_v1.0.0.md` - This summary document
