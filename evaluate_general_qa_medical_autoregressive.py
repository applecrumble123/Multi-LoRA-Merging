#from huggingface_hub import notebook_login
#from google.colab import userdata
import os
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, BertTokenizer, BertModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
import torch
from collections import defaultdict
import jsonlines as jl
from peft import LoraConfig, TaskType, PeftModel
#from peft import LoraConfig, TaskType
#from new_peft.src.peft import PeftModel, LoraConfig, TaskType
import torch.nn as nn
import copy
import evaluate
import re
from tqdm import tqdm

import gc

import gensim.downloader as api
import numpy as np
from scipy.spatial.distance import cosine
from evaluate import load  # Import the Hugging Face evaluation library
import io
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)  # Set your desired seed here

# Define device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

### define the parameters
model_name = "google/gemma-2b"
new_model = "gemma-ft"

hf_token = "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"

# Load the entire model on the GPU 0
# device_map = {"": 0}
# "auto" for multple GPUs
device_map = "auto"
#device_map = {"": 0}

#root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop"
root_folder = "/home/ums/Desktop"
root_code_folder  = os.path.join(root_folder,"add_multiple_lora_to_base_model")

medical_qa_test_path = os.path.join(root_code_folder, "dataset/medical_qa/val.jsonl")

base_model_weights_path = os.path.join(root_code_folder, "gemma_2b_base_model.pt")
#merged_peft_weights_medical_qa_path = os.path.join(root_code_folder, "Medical QA Weights/merged_peft_medical_qa_lora.pt")
combined_weights_medical_qa_path = os.path.join(root_code_folder, "Medical QA Weights/combined_model_with_filtered_medical_qa_lora.pt")
combined_weights_general_qa_path = os.path.join(root_code_folder, "General QA Weights/combined_model_with_filtered_general_qa_lora.pt")
combined_weights_medical_autoregressive_path = os.path.join(root_code_folder, "Medical autoregressive weights/combined_model_with_filtered_medical_autoregressive_lora.pt")
combined_weights_general_qa_medical_autoregressive_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_model_with_general_qa_medical_autoregressive_lora.pt")


################ results file paths

base_general_qa_generated_qns_text_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/base_general_qa_model_generated_answers.txt")
base_general_qa_results_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/base_general_qa_model_results.txt")

#merged_peft_medical_qa_generated_qns_text_path = os.path.join(root_code_folder, "dataset/merged_peft_medical_qa_generated_answers.txt")
#merged_peft_medical_qa_results_path = os.path.join(root_code_folder, "dataset/merged_peft_medical_qa_model_results.txt")

combined_medical_qa_generated_qns_text_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_medical_qa_generated_answers.txt")
combined_medical_qa_results_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_medical_qa_model_results.txt")

combined_general_qa_generated_qns_text_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_general_qa_generated_answers.txt")
combined_general_qa_results_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_general_qa_model_results.txt")

combined_medical_autoregressive_generated_qns_text_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_medical_autoregressive_generated_answers.txt")
combined_medical_autoregressive_results_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_medical_autoregressive_model_results.txt")

combined_general_qa_medical_autoregressive_generated_qns_text_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_general_qa_medical_autoregressive_generated_answers.txt")
combined_general_qa_medical_autoregressive_results_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_general_qa_medical_autoregressive_model_results.txt")

base_model_weights = torch.load(base_model_weights_path, map_location="cuda:3")
combined_weights_medical_qa_weights = torch.load(combined_weights_medical_qa_path, map_location="cuda:3")
#merged_peft_weights_medical_qa_weights = torch.load(merged_peft_weights_medical_qa_path, map_location="cuda:3")
combined_weights_general_qa_weights = torch.load(combined_weights_general_qa_path, map_location="cuda:3")
combined_weights_medical_autoregressive_weights = torch.load(combined_weights_medical_autoregressive_path, map_location="cuda:3")
combined_weights_general_qa_medical_autoregressive_weights = torch.load(combined_weights_general_qa_medical_autoregressive_path, map_location="cuda:3")

# extract the test data
def get_test_data(test_json_path):
    test_data = []
    with jl.open(test_json_path, 'r') as reader:
        for data in reader:
            # Get "instruction" and "response" fields directly
            instruction = data.get("instruction", "")
            response = data.get("response", "Unanswerable")  # Default to "Unanswerable" if no response

            # Format as "Instruction" and "Response" for the filtered dataset
            formatted_text = "Instruction:\n{}\n\nResponse:\n{}".format(instruction, response)
            test_data.append({"text": formatted_text})
    return test_data

medical_qa_test_data = get_test_data(medical_qa_test_path)
medical_qa_test_data = medical_qa_test_data[:100]

# # Convert test data to Hugging Face Dataset format (for compatibility with metrics)
# test_dataset = Dataset.from_list(test_data)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)

# end of sequence token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # fix weird overflow issue with fp16 training

# Load the ROUGE metric
rouge = load("rouge")

# Define a function to calculate ROUGE scores for generated answers
def calculate_rouge(reference, generated):
    return rouge.compute(predictions=[generated], references=[reference])

# Function to generate answers for General QA
def generate_medical_answer(question, model, tokenizer, max_length=40, device="cuda:3"):
    
    torch.manual_seed(42)

    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=60,
    #         temperature=0.5,
    #         top_k=20,
    #         top_p=0.7,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=350,  # Increase max length for complex answers
            num_return_sequences=1,
            top_k=50,
            top_p=0.85,  # Narrow top-p for more deterministic output
            temperature=0.3,  # Slightly higher temperature for balance between creativity and accuracy
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs = model.generate(
    #     **inputs,
    #     max_length=350,  # Increase max length for complex answers
    #     num_return_sequences=1,
    #     top_k=50,
    #     top_p=0.85,  # Narrow top-p for more deterministic output
    #     temperature=0.3,  # Slightly higher temperature for balance between creativity and accuracy
    #     no_repeat_ngram_size=3,
    # )

    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the question from the answer if it appears
    answer = answer.split(question)[-1].strip() if question in answer else answer.strip()
    return answer



# Modify the main evaluation function to use the appropriate generation function
def generate_answer_and_rouge(question, reference_answer, model, tokenizer):
    generated_answer = generate_medical_answer(question, model, tokenizer)
    rouge_scores = calculate_rouge(reference_answer, generated_answer)
    return generated_answer, rouge_scores

# Main evaluation function that evaluates with ROUGE scoring based on task specificity
def evaluate_model_with_rouge(model_weights_path, test_data, output_file_path, generated_answers_file_path, tokenizer, model_name="gemma-2b", device="cuda:3"):
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, torch_dtype=torch.float16)
    # Check if model_weights is already an OrderedDict (pre-loaded weights)
    if isinstance(model_weights_path, dict):
        model.load_state_dict(model_weights_path)
    else:
        # Otherwise, assume it is a file path and use torch.load
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    with open(generated_answers_file_path, "w") as gen_file:
        for i, item in enumerate(test_data):
            text = item.get('text', '')
            parts = text.split("\n\nResponse:\n")
            if len(parts) == 2:
                question = parts[0].replace("Instruction:\n", "").strip()
                reference_answer = parts[1].strip()
            else:
                print(f"Unexpected format in item: {text}")
                continue

            generated_answer, rouge_scores = generate_answer_and_rouge(question, reference_answer, model, tokenizer)
            gen_file.write(f"Question {i+1}: {question}\nGenerated Answer {i+1}: {generated_answer}\nReference Answer {i+1}: {reference_answer}\n\n")

            rouge1_scores.append(rouge_scores["rouge1"])
            rouge2_scores.append(rouge_scores["rouge2"])
            rougeL_scores.append(rouge_scores["rougeL"])

    avg_rouge_scores = {"rouge1": np.mean(rouge1_scores), "rouge2": np.mean(rouge2_scores), "rougeL": np.mean(rougeL_scores)}
    rouge_std_dev = {"rouge1": np.std(rouge1_scores), "rouge2": np.std(rouge2_scores), "rougeL": np.std(rougeL_scores)}

    output_text = (
        "Individual ROUGE Scores:\n"
        "ROUGE-1: " + str(rouge1_scores) + "\n"
        "ROUGE-2: " + str(rouge2_scores) + "\n"
        "ROUGE-L: " + str(rougeL_scores) + "\n"
        "Average ROUGE Scores: " + str(avg_rouge_scores) + "\n"
        "Standard Deviation of ROUGE Scores: " + str(rouge_std_dev) + "\n"
    )
    with open(output_file_path, "w") as file:
        file.write(output_text)

    print(f"Results saved to {output_file_path}")
    print(f"Questions and generated answers saved to {generated_answers_file_path}")
    return avg_rouge_scores, rouge_std_dev

# Evaluation function for General QA with ROUGE scoring
def evaluate_medical_qa(model_weights_path, test_data, output_file_path, generated_answers_file_path, tokenizer, model_name="gemma-2b", device="cuda:3"):
    return evaluate_model_with_rouge(
        model_weights_path, test_data, output_file_path, generated_answers_file_path, tokenizer, model_name, device
    )

# Set a fixed seed for reproducibility
set_seed(42)

# Example usage for the General QA and Medical QA models
medical_qa_base_model_avg_rouge_scores, medical_qa_base_model_rouge_std_dev = evaluate_medical_qa(
    model_weights_path=base_model_weights,
    test_data=medical_qa_test_data,
    output_file_path=base_general_qa_results_path,
    generated_answers_file_path=base_general_qa_generated_qns_text_path,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device
)

# merged_peft_medical_qa_lora_avg_rouge_scores, merged_peft_medical_qa_lora_rouge_std_dev = evaluate_medical_qa(
#     model_weights_path=merged_peft_weights_medical_qa_weights,
#     test_data=medical_qa_test_data,
#     output_file_path=merged_peft_medical_qa_results_path,
#     generated_answers_file_path=merged_peft_medical_qa_generated_qns_text_path,
#     tokenizer=tokenizer,
#     model_name=model_name,
#     device=device
# )

combined_medical_qa_lora_avg_rouge_scores, combined_medical_qa_lora_rouge_std_dev = evaluate_medical_qa(
    model_weights_path=combined_weights_medical_qa_weights,
    test_data=medical_qa_test_data,
    output_file_path=combined_medical_qa_results_path,
    generated_answers_file_path=combined_medical_qa_generated_qns_text_path,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device
)

combined_general_qa_lora_avg_rouge_scores, combined_general_qa_lora_rouge_std_dev = evaluate_medical_qa(
    model_weights_path=combined_weights_general_qa_weights,
    test_data=medical_qa_test_data,
    output_file_path=combined_general_qa_results_path,
    generated_answers_file_path=combined_general_qa_generated_qns_text_path,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device
)

combined_medical_autoregressive_lora_avg_rouge_scores, combined_medical_autoregressive_lora_rouge_std_dev = evaluate_medical_qa(
    model_weights_path=combined_weights_medical_autoregressive_weights,
    test_data=medical_qa_test_data,
    output_file_path=combined_medical_autoregressive_results_path,
    generated_answers_file_path=combined_medical_autoregressive_generated_qns_text_path,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device
)

combined_general_qa_medical_autoregressive_lora_avg_rouge_scores, combined_general_qa_medical_autoregressive_lora_rouge_std_dev = evaluate_medical_qa(
    model_weights_path=combined_weights_general_qa_medical_autoregressive_weights,
    test_data=medical_qa_test_data,
    output_file_path=combined_general_qa_medical_autoregressive_results_path,
    generated_answers_file_path=combined_general_qa_medical_autoregressive_generated_qns_text_path,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device
)

# Print results
print("\n===================== Base Model Medical QA ==================\n")
print("Medical QA Base Model Average ROUGE Scores:", medical_qa_base_model_avg_rouge_scores)
print("Medical QA Base Model Standard Deviation:", medical_qa_base_model_rouge_std_dev)

# print("\n-------------------------- Merged Peft Medical QA -----------------------------------\n")
# print("Medical QA Peft LoRA Model Average ROUGE Scores:", merged_peft_medical_qa_lora_avg_rouge_scores)
# print("Medical QA Peft LoRA Model Standard Deviation:", merged_peft_medical_qa_lora_rouge_std_dev)

print("\n--------------------------- Custom Medical QA Combined ----------------------------------------\n")
print("Medical QA Custom LoRA Model Average ROUGE Scores:", combined_medical_qa_lora_avg_rouge_scores)
print("Medical QA Custom LoRA Model Standard Deviation:", combined_medical_qa_lora_rouge_std_dev)

print("\n--------------------------- Custom General QA Combined ----------------------------------------\n")
print("General QA Custom LoRA Model Average ROUGE Scores:", combined_general_qa_lora_avg_rouge_scores)
print("General QA Custom LoRA Model Standard Deviation:", combined_general_qa_lora_rouge_std_dev)

print("\n--------------------------- Custom Medical Autoregressive Combined ----------------------------------------\n")
print("Medical Autoregressive Custom LoRA Model Average ROUGE Scores:", combined_medical_autoregressive_lora_avg_rouge_scores)
print("Medical Autoregressive Custom LoRA Model Standard Deviation:", combined_medical_autoregressive_lora_rouge_std_dev)

print("\n--------------------------- Custom General QA + Medical Autoregressive Combined ----------------------------------------\n")
print("General QA + Medical Autoregressive Custom LoRA Model Average ROUGE Scores:", combined_general_qa_medical_autoregressive_lora_avg_rouge_scores)
print("General QA + Medical Autoregressive Custom LoRA Model Standard Deviation:", combined_general_qa_medical_autoregressive_lora_rouge_std_dev)




