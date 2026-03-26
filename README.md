# PyTorch to TFLite: The 2026 Pain-Free Conversion Guide for Android Edge AI

A streamlined, tested pipeline to train a Hugging Face Transformer (DistilBERT) on a custom CSV dataset and convert it directly to a `.tflite` model optimized for local Android inference. 

## 🚨 The Problem: Dependency Hell
If you have ever tried to train a Transformer model on a local machine without a dedicated GPU (e.g., using Intel Iris graphics) and convert it to TensorFlow Lite for mobile deployment, you likely ran into:
1. **Slow Training:** Integrated graphics take hours or days to train even small transformer models.
2. **Version Conflicts:** Converting PyTorch -> ONNX -> TensorFlow -> TFLite on a local Windows machine often results in deep-level dependency crashes, especially on Python 3.11+.
3. **Data Type Mismatches:** Android's TFLite libraries strictly require `INT64` (Long) buffer types for text tokens, while standard Python converters default to `INT32`, causing immediate app crashes.

## 💡 The Solution: Colab T4 GPU + LiteRT
This repository provides a clean, web-based pipeline using **Google Colab** to bypass local hardware limits and environment errors completely.
* **Speed:** We use Colab's free **T4 GPU** with mixed precision (`fp16`) to train the custom CSV dataset into `.safetensors` in minutes.
* **Stability:** We use Google's modern `litert-torch` library to go *directly* from PyTorch to Edge TFLite, completely bypassing buggy ONNX/TensorFlow intermediate bridges.

---

## 📂 Repository Structure

```text
pytorch-to-android-tflite/
├── LICENSE                     # MIT License
├── README.md                   # You are here
├── requirements.txt            # Python dependencies
├── 1_train_model.py            # Phase 1: Custom dataset to PyTorch weights
├── 2_convert_tflite.py         # Phase 2: PyTorch to Edge TFLite
├── 3_extract_vocab.py          # Phase 3: Android dictionary extraction
└── 4_verify_model.py           # Phase 4: Mock inference & INT64 validation
