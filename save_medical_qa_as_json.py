import os
import json
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel
import jsonlines as jl

# Paths
root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/"
original_train_json_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/medical_qa/train.json")

# Paths for both JSON and JSON Lines formats
saved_train_json_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/medical_qa/new_train_data.json")
saved_val_json_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/medical_qa/new_val_data.json")
train_jsonl_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/medical_qa/new_train_data.jsonl")
val_jsonl_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/medical_qa/new_val_data.jsonl")

# Load newline-delimited JSON file
medical_qa_data = []
with open(original_train_json_path, "r", encoding="utf-8") as f:
    for line in f:
        medical_qa_data.append(json.loads(line.strip()))

# Transform data to instruction-response format
instruction_response_data = []
for item in medical_qa_data:
    instruction = item.get("question", "")
    response = item.get("exp") or "Unanswerable"  # Default to "Unanswerable" if "exp" is missing

    instruction_response_data.append({
        "instruction": instruction,
        "response": response
    })

# Shuffle data
random.shuffle(instruction_response_data)

# Split data: 90% for training, 10% for validation
split_index = int(len(instruction_response_data) * 0.9)
train_data = instruction_response_data[:split_index]
val_data = instruction_response_data[split_index:]

# Save to JSON files (array format)
with open(saved_train_json_path, "w", encoding="utf-8") as train_file:
    json.dump(train_data, train_file, indent=4)

with open(saved_val_json_path, "w", encoding="utf-8") as val_file:
    json.dump(val_data, val_file, indent=4)

# Save training data to JSON Lines format
with open(train_jsonl_path, "w", encoding="utf-8") as train_jsonl_file:
    for entry in train_data:
        json.dump(entry, train_jsonl_file)
        train_jsonl_file.write("\n")

# Save validation data to JSON Lines format
with open(val_jsonl_path, "w", encoding="utf-8") as val_jsonl_file:
    for entry in val_data:
        json.dump(entry, val_jsonl_file)
        val_jsonl_file.write("\n")

print("Data saved to JSON and JSON Lines formats.")
