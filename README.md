# PyTorch to TFLite: The Pain-Free Conversion Guide for Android Edge AI

A streamlined, tested pipeline to train a Hugging Face Transformer (DistilBERT) on a custom CSV dataset and convert it directly to a `.tflite` model optimized for an Android app. 

## 🚨 The Problem: Dependency Hell
If you have ever tried to train a Transformer model on a local machine without a dedicated GPU (e.g., using Intel Iris graphics) and convert it to TensorFlow Lite for mobile deployment, you likely ran into:
1. **Slow Training:** Integrated graphics take hours/days to train even small transformer models.
2. **Version Conflicts:** Converting PyTorch -> ONNX -> TensorFlow -> TFLite on a local Windows machine often results in deep-level dependency crashes, especially on Python 3.11+.
3. **Data Type Mismatches:** TFLite on Android requires `INT64` buffer types, while standard Python converters default to `INT32`.

## 💡 The Solution: Colab T4 GPU + LiteRT
This repository provides a clean, web-based pipeline using **Google Colab** to bypass local hardware limits and environment errors completely.
* **Speed:** We use Colab's free **T4 GPU** to train the custom CSV dataset into `.safetensors` in minutes.
* **Stability:** We use Google's modern `litert-torch` library to go *directly* from PyTorch to TFLite, completely bypassing the buggy ONNX/TensorFlow intermediate bridges.

---

## 📂 Repository Structure

* `1_train_model.ipynb` - Script to train your custom CSV dataset to `model.safetensors`.
* `2_convert_tflite.ipynb` - Script to convert the PyTorch model directly to `model.tflite`.
* `3_extract_vocab.py` - Helper script to extract the `vocab.txt` dictionary required for Android.

---

## 🚀 Step-by-Step Guide

### Step 1: Train the Model (CSV to Safetensors)
*Open `1_train_model.ipynb` in Google Colab and ensure your runtime is set to **T4 GPU**.*

1. Upload your custom dataset as a CSV file (e.g., `dataset.csv` with `text` and `label` columns).
2. The script uses Hugging Face's `Trainer` API with `fp16=True` to maximize the T4 GPU's speed.
3. The output will be a highly accurate DistilBERT model saved as `model.safetensors` alongside its configuration files.

### Step 2: Convert to Edge Format (Safetensors to TFLite)
*Open `2_convert_tflite.ipynb` in Google Colab.*

Instead of using legacy conversion tools, this script uses Google's latest edge conversion engine:
1. It loads your trained `model.safetensors`.
2. It traces the model using a dummy input of `(1, 128)` with `dtype=torch.long` (ensuring the `INT64` requirement for Android is met).
3. It uses `litert_torch.convert()` to generate a highly optimized `model.tflite` file.

### Step 3: Extract the Android Dictionary (Vocab.txt)
Modern Transformers use "Fast Tokenizers" that output a `.json` file, but Android's TFLite libraries natively expect a flat `vocab.txt` file to translate words into integer IDs. 

Run `3_extract_vocab.py` to iterate through the tokenizer's internal memory and generate a clean, line-by-line `vocab.txt` mapping.

---

## 🎯 Final Outputs
After running this pipeline, you will have exactly what your Android app needs to run AI inference offline:
1. **`model.tflite`**: The compiled, edge-optimized brain.
2. **`vocab.txt`**: The dictionary mapping words to model IDs.
3. **`labels.txt`**: A manual text file you create to map your integer outputs (0, 1, 2) to your actual category names.

*(Simply drop these three files into your Android project's `assets` folder to begin local, on-device classification).*

---

## 📝 License
This project is licensed under the MIT License - feel free to use, modify, and distribute this pipeline for your own edge AI projects.
