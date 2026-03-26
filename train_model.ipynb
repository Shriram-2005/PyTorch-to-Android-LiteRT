"""
Phase 1: Train Custom Dataset to PyTorch Weights (.safetensors)
---------------------------------------------------------------
This script fine-tunes a DistilBERT transformer on a custom CSV dataset.
It is highly optimized to run on a Google Colab T4 GPU (using fp16 precision 
and optimal batch sizes) to reduce training time from hours to minutes.
"""

import os
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ---------------------------------------------------------
# 1. Verify Hardware Acceleration
# ---------------------------------------------------------
if torch.cuda.is_available():
    print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CPU is active! For faster training, go to Runtime > Change runtime type > Select T4 GPU.")

# ---------------------------------------------------------
# 2. Load and Prepare the Custom Dataset
# ---------------------------------------------------------
# Replace 'custom_dataset.csv' with your file. 
# Expected format: two columns named 'text' (string) and 'label' (integers: 0, 1, 2...)
df = pd.read_csv("custom_dataset.csv")
df['label'] = df['label'].astype(int)

# Split 90% for training, 10% for validation
hf_dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1)
print(f"✅ Training samples: {len(hf_dataset['train'])} | Validation samples: {len(hf_dataset['test'])}")

# ---------------------------------------------------------
# 3. Load Tokenizer
# ---------------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ---------------------------------------------------------
# 4. Tokenization Function
# ---------------------------------------------------------
# Note: max_length=128 is heavily recommended for mobile Edge AI. 
# It keeps memory usage low and inference speeds fast on Android devices.
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("Tokenizing datasets...")
tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------------------------------------
# 5. Define Evaluation Metric
# ---------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# ---------------------------------------------------------
# 6. Load the Untrained Model Architecture
# ---------------------------------------------------------
# Update num_labels and the mapping to match your specific dataset categories.
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={0: "Category_A", 1: "Category_B", 2: "Category_C"},
    label2id={"Category_A": 0, "Category_B": 1, "Category_C": 2}
)

# ---------------------------------------------------------
# 7. T4-Optimized Training Arguments
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./distilbert_custom_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    per_device_train_batch_size=64, # Optimized to max out 16GB T4 VRAM
    per_device_eval_batch_size=64,
    fp16=True,                      # Enables mixed-precision for massive speedup on T4
    dataloader_num_workers=2,
    optim="adamw_torch",
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none"
)

# ---------------------------------------------------------
# 8. Initialize Trainer
# ---------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------
# 9. Train the Model
# ---------------------------------------------------------
print("\n• Starting Training on T4 GPU...")
trainer.train()

# ---------------------------------------------------------
# 10. Save Final Weights
# ---------------------------------------------------------
# This folder will contain the config.json and model.safetensors needed for Phase 2.
final_model_path = "./saved_model_weights"
trainer.save_model(final_model_path)
print(f"\n✅ Training Complete! Model weights saved to: {final_model_path}")
