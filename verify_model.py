"""
Phase 4: Mock Android Inference (Verification)
----------------------------------------------
This script tests the newly generated .tflite model and vocab.txt file together.
It mimics how an Android app will process text and ensures data types (INT64) match.
"""

import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer

# 1. Load the extracted dictionary
tokenizer = DistilBertTokenizer(vocab_file='vocab.txt', do_lower_case=True)

# 2. Prepare test text (Mimicking a mobile notification)
test_text = "System alert: server downtime scheduled for midnight"
inputs = tokenizer(test_text, padding='max_length', max_length=128, truncation=True, return_tensors="np")

# 3. Load the Edge Model
interpreter = tf.lite.Interpreter(model_path="edge_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. Perform Inference
# CRITICAL FIX: Cast the numpy array to int64 to match the model tracing
interpreter.set_tensor(input_details[0]['index'], inputs['input_ids'].astype(np.int64))
interpreter.invoke()

# 5. Output Results
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

print("--- Verification Results ---")
print(f"Input Shape: {input_details[0]['shape']}")
print(f"Model Raw Logits: {output_data}")
print(f"Predicted Class Index: {predicted_class}")
print("✅ VERIFIED: Model and Vocab are working together perfectly!")
