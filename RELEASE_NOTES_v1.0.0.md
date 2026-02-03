# FastVLM v1.0.0 Release Notes

**Release Date:** January 2025  
**Repository:** [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)

## Overview

FastVLM v1.0.0 is the initial release of the FastVLM codebase, providing efficient vision encoding for Vision Language Models. This release includes the complete implementation of FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images.

## Key Features

### 🚀 Core Model Implementation
- **FastViTHD Vision Encoder**: Novel hybrid vision encoder architecture
- **Multiple Model Variants**: 0.5B, 1.5B, and 7B parameter models
- **Stage 2 & Stage 3 Checkpoints**: Pre-trained models available for all variants
- **Qwen2 LLM Integration**: Built on Qwen2 language models for superior performance

### 📊 Performance Highlights
- **85x faster TTFT** (Time-to-First-Token) compared to LLaVA-OneVision-0.5B
- **3.4x smaller vision encoder** than comparable models
- **7.9x faster TTFT** for 7B variants compared to Cambrian-1-8B
- Outperforms recent works while using a single image encoder

### 🔧 Inference Capabilities
- **PyTorch Inference**: Full support for PyTorch model inference
- **CLI Tool**: `predict.py` for easy command-line inference
- **Flexible Prompting**: Support for custom prompts and conversation templates
- **Multi-Image Support**: Process single or multiple images
- **Configurable Generation**: Temperature, top-p, beam search parameters

### 🍎 Apple Silicon Support
- **MLX Export Tools**: Convert models for Apple Silicon inference
- **CoreML Vision Encoder Export**: Export vision encoder using CoreML
- **Quantization Support**: 4-bit, 8-bit, and FP16 quantization options
- **Pre-exported Models**: Ready-to-use models for Apple Silicon (0.5B, 1.5B, 7B)

### 📱 iOS & macOS Application
- **Native Swift App**: Full-featured iOS/macOS application
- **Real-time Inference**: On-device model inference with TTFT metrics
- **Camera Integration**: Live camera feed processing
- **Flexible Prompts**: Built-in prompt library with customization
- **Privacy-First**: All processing done on-device

### 🌐 Serving Infrastructure
- **Gradio Web Server**: Web-based interface for model serving
- **Model Workers**: Distributed model serving architecture
- **Controller/Worker Pattern**: Scalable serving infrastructure
- **SGLang Integration**: Support for SGLang-based serving

### 🎓 Training Support
- **LLaVA Training Codebase**: Full training pipeline integration
- **DeepSpeed Support**: Distributed training capabilities
- **Qwen2 Training**: Specialized training scripts for Qwen2 models
- **Custom Training**: Support for fine-tuning and custom variants

## Model Zoo

| Model        | Stage | PyTorch Checkpoint |
|:-------------|:-----:|:------------------:|
| FastVLM-0.5B |   2   | [Download](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip) |
|              |   3   | [Download](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) |
| FastVLM-1.5B |   2   | [Download](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip) |
|              |   3   | [Download](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip) |
| FastVLM-7B   |   2   | [Download](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip) |
|              |   3   | [Download](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip) |

## Technical Specifications

### Dependencies
- Python 3.8+
- PyTorch 2.6.0
- Transformers 4.48.3
- CoreML Tools 8.2
- Gradio 5.11.0
- FastAPI & Uvicorn

### Architecture Components
- **Vision Encoder**: FastViTHD (MobileCLIP-based)
- **Language Model**: Qwen2 (0.5B, 1.5B, 7B variants)
- **Multimodal Projector**: Custom projection layer
- **Conversation Templates**: Qwen2 conversation format

### Codebase Statistics
- **39 Python files**: Core model, training, and serving code
- **10 Swift files**: iOS/macOS application
- **Full LLaVA Integration**: Compatible with LLaVA training pipeline

## Usage Examples

### Basic Inference
```bash
python predict.py --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
                  --image-file image.png \
                  --prompt "Describe the image."
```

### Model Export for Apple Silicon
```bash
# Export vision encoder
python model_export/export_vision_encoder.py --model-path /path/to/checkpoint

# Export to MLX format
python -m mlx_vlm.convert --hf-path /path/to/checkpoint \
                          --mlx-path /path/to/exported \
                          --only-llm -q --q-bits 8
```

### iOS App Setup
```bash
chmod +x app/get_pretrained_mlx_model.sh
app/get_pretrained_mlx_model.sh --model 0.5b --dest app/FastVLM/model
```

## Documentation

- **Main README**: Comprehensive setup and usage guide
- **Model Export Guide**: Detailed Apple Silicon export instructions
- **iOS App README**: Complete app setup and customization guide
- **Paper**: [FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303)

## Research & Citation

This release corresponds to the CVPR 2025 paper:

```bibtex
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## License

- **Code License**: See [LICENSE](LICENSE)
- **Model License**: See [LICENSE_MODEL](LICENSE_MODEL)

## Acknowledgments

This codebase builds upon multiple open-source contributions. See [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for complete details.

## What's Next

This initial release provides a complete foundation for:
- Research and experimentation with efficient vision-language models
- Production deployment on Apple Silicon devices
- Custom model training and fine-tuning
- Integration into larger vision-language systems

---

**Full Changelog**: Initial release - v1.0.0
