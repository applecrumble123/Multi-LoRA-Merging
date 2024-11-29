import os
import csv
import json
import random
import re

# Define paths
input_csv_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/medical_qa_csv"  # Folder containing your CSV files
output_train_json_path = "/mnt/c/Users/tohji/OneDrive/Desktop/add_multiple_lora_to_base_model/dataset/medical_qa/train.jsonl"  # Single training JSON Lines file
output_val_json_path = "/mnt/c/Users/tohji/OneDrive/Desktop/add_multiple_lora_to_base_model/dataset/medical_qa/val.jsonl"  # Single validation JSON Lines file

# Initialize empty lists for training and validation data
train_data = []
val_data = []

# Function to clean response text
def clean_text(text):
    # Remove tabs and newlines, collapse multiple spaces into one
    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text.strip()

# Function to process a single CSV file
def process_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)  # Automatically reads headers
        for row in reader:
            question = row.get("Question", "").strip()
            answer = row.get("Answer", "Unanswerable").strip()
            answer = clean_text(answer)  # Clean response text
            data.append({"instruction": question, "response": answer})
    return data

# Process all CSV files in the input folder
for filename in os.listdir(input_csv_folder):
    if filename.endswith(".csv"):
        input_file_path = os.path.join(input_csv_folder, filename)

        # Read and process the CSV file
        data = process_csv(input_file_path)
        random.shuffle(data)  # Shuffle data for randomness

        # Split into 90% train and 10% validation
        split_index = int(len(data) * 0.9)
        train_data.extend(data[:split_index])
        val_data.extend(data[split_index:])

# Save training data to a single JSON Lines file
with open(output_train_json_path, "w", encoding="utf-8") as train_file:
    for entry in train_data:
        json.dump(entry, train_file)
        train_file.write("\n")

# Save validation data to a single JSON Lines file
with open(output_val_json_path, "w", encoding="utf-8") as val_file:
    for entry in val_data:
        json.dump(entry, val_file)
        val_file.write("\n")

print(f"Training data saved to {output_train_json_path}")
print(f"Validation data saved to {output_val_json_path}")
