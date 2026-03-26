"""
Phase 2: Direct LiteRT (TFLite) Conversion
------------------------------------------
This script bypasses legacy ONNX and TensorFlow bridges. 
It uses Google's `litert-torch` to convert PyTorch weights directly into 
an edge-optimized .tflite model for Android.
"""

import torch
import litert_torch
from transformers import DistilBertForSequenceClassification

# ---------------------------------------------------------
# 1. Load the Trained PyTorch Weights
# ---------------------------------------------------------
# We explicitly set num_labels to 3 to match the custom training phase.
# If this doesn't match your dataset, the Android app will crash on inference.
print("--- Loading trained weights ---")
model_path = './saved_model_weights' # Path from Phase 1
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=3)
model.eval()

# ---------------------------------------------------------
# 2. Prepare the Dummy Input for Tracing
# ---------------------------------------------------------
# CRITICAL: We use dtype=torch.long (INT64) here. 
# Android's TFLite interpreter natively expects INT64 for input IDs. 
# Using the default INT32 will cause a ValueError in Android Studio.
# Batch Size: 1 | Sequence Length: 128 (Optimized for mobile)
sample_inputs = (torch.randint(0, 30522, (1, 128), dtype=torch.long),)

# ---------------------------------------------------------
# 3. Execute Direct LiteRT Conversion
# ---------------------------------------------------------
print("--- Starting Direct LiteRT Conversion ---")
try:
    # Use Google's direct conversion engine
    edge_model = litert_torch.convert(model, sample_inputs)
    
    # Export as the final .tflite flatbuffer
    output_filename = 'edge_model.tflite'
    edge_model.export(output_filename)
    
    print(f"🎉 SUCCESS! Edge model ready: {output_filename}")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
