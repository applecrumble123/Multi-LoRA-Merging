import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel
import jsonlines as jl

root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/"
train_json_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/general_qa/general_qa_train_data.jsonl")
val_json_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/dataset/general_qa/general_qa_val_data.jsonl")

# Load SQuAD v2 dataset
squad_v2 = load_dataset("squad_v2")

# Helper function to convert data to instruction-response format
def convert_to_instruction_response(data_split):
    data = []
    for item in data_split:
        question = item["question"]
        answers = item["answers"]["text"]
        response = answers[0] if answers else "Unanswerable"  # Use the first answer or "Unanswerable"

        # Format each entry as an instruction-response pair
        entry = {
            "instruction": question,
            "response": response
        }
        data.append(entry)
    return data

# Convert train and validation splits
train_data = convert_to_instruction_response(squad_v2["train"])
validation_data = convert_to_instruction_response(squad_v2["validation"])

# Save train data to JSON Lines format
with open(train_json_path, "w", encoding="utf-8") as train_file:
    for entry in train_data:
        json.dump(entry, train_file)
        train_file.write("\n")

# Save validation data to JSON Lines format
with open(val_json_path, "w", encoding="utf-8") as val_file:
    for entry in validation_data:
        json.dump(entry, val_file)
        val_file.write("\n")

print("Data saved to general_qa_train_data.jsonl and general_qa_val_data.jsonl")
