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
import json
import gc
import gensim.downloader as api
import numpy as np
from scipy.spatial.distance import cosine
from evaluate import load  # Import the Hugging Face evaluation library

# Define device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

### define the parameters
model_name = "google/gemma-2b"
new_model = "gemma-ft"

hf_token = "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"

# Load the entire model on the GPU 0
# device_map = {"": 0}
# "auto" for multple GPUs
#device_map = "auto"
device_map = {"": 3}

#root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop"
root_folder = "/home/ums/Desktop"
root_code_folder  = os.path.join(root_folder,"add_multiple_lora_to_base_model")

base_model_weights_path = os.path.join(root_code_folder, "gemma_2b_base_model.pt")
general_qa_lora_path = os.path.join(root_code_folder, "General QA Weights/general_qa_lora.pt")
combined_model_with_filtered_general_qa_lora_path = os.path.join(root_code_folder, "General QA Weights/combined_model_with_filtered_general_qa_lora.pt")

combined_general_qa_lora_param_difference_file_path = os.path.join(root_code_folder, "General QA Weights/combined_general_qa_lora_param_difference.txt")

# base_model_weights = torch.load(base_model_weights_path, map_location=device)
general_qa_lora_weights = torch.load(general_qa_lora_path, map_location=device)

# for name, weight in lora_model_weights.items():
#     print(f"{name}: shape = {weight.shape}")

# Print the names and shapes of the saved weights
# print("Saved Base Model weights:")
# for name, weight in base_model_weights.items():
#     print(f"{name}: shape = {weight.shape}")



# # Display filtered LoRA weights
# print("Filtered LoRA weights:")
# for name, weight in filtered_lora_weights.items():
#     print(f"{name}: shape = {weight.shape}")


filtered_general_qa_lora_weights = torch.load(general_qa_lora_path, map_location=device)

# print("Keys in filtered_general_qa_lora_weights:")
# for key in filtered_general_qa_lora_weights.keys():
#     print(key)



target_modules = ['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj']

def combine_lora_and_base(loaded_base_model_weights, filtered_lora_weights, save_path, alpha, rank, save:bool = True):
    for name, base_weight in loaded_base_model_weights.items():
        matched = False
        for target in target_modules:
            if f".{target}." in name:
                matched = True
                print(f"Processing {name} in target module '{target}'")
                
                # Construct the expected LoRA weight keys
                base_name_no_suffix = name.replace(".weight", "")
                lora_a_key = f"base_model.model.{base_name_no_suffix}.lora_A.default.weight"
                lora_b_key = f"base_model.model.{base_name_no_suffix}.lora_B.default.weight"
                
                # Check if LoRA keys exist and log their presence
                if lora_a_key in filtered_lora_weights and lora_b_key in filtered_lora_weights:
                    print(f"Found matching LoRA weights: {lora_a_key} and {lora_b_key}")

                    # Extract LoRA weights
                    lora_A = filtered_lora_weights[lora_a_key]
                    lora_B = filtered_lora_weights[lora_b_key]
                    
                    print("lora_A shape: ", lora_A.shape)
                    print("lora_B shape: ", lora_B.shape)

                    # Compute the low-rank update term (A * B^T)
                    delta_weight = lora_B @ lora_A
                    print(f"Computed scaling factor with shape: {delta_weight.shape}")

                    # Scale the base weight by the computed scaling factor
                    # Ensure the shapes match for element-wise multiplication
                    if delta_weight.shape == base_weight.shape:
                        # Apply the scaling formula: w = w + w * (A * B^T)
                        #base_model_weights_new[name] = base_weight + base_weight * delta_weight
                        loaded_base_model_weights[name] = loaded_base_model_weights[name]  + (alpha/rank) * delta_weight
                        #base_model_weights_new[name] = base_weight + base_weight * scaling_factor
                    else:
                        #base_model_weights_new[name] = base_weight
                        print(f"Shape mismatch for {name}: base weight shape {base_weight.shape} vs scaling factor shape {delta_weight.shape}")
                else:
                    print(f"LoRA weights not found for {name}. Skipping update.")
                break
        
        # Ensure skipping message prints only once if no match
        if not matched:
            print(f"Skipping {name}, not in target modules.")

    # Now, create a new state_dict for saving
    # updated_state_dict = base_model_weights.copy()
    # updated_state_dict.update(base_model_weights_new)
    if save == True:
        torch.save(loaded_base_model_weights, save_path)

base_model_weights_general_qa = torch.load(base_model_weights_path, map_location=device)

combine_lora_and_base(base_model_weights_general_qa, filtered_general_qa_lora_weights, combined_model_with_filtered_general_qa_lora_path, alpha=128, rank=64,save = True)
combined_model_with_filtered_general_qa_lora_weights = torch.load(combined_model_with_filtered_general_qa_lora_path, map_location=device)

# def compare_model_parameters_with_percentage_to_file(base_model, lora_model, output_file, epsilon=1e-10):
#     """
#     Compare parameters between a base model and a combined model (base + LoRA),
#     calculate percentage differences, and save results to a file.
    
#     Args:
#         base_model (OrderedDict): State dict of the base model.
#         lora_model (OrderedDict): State dict of the combined model.
#         output_file (str): Path to save the results.
#         epsilon (float): Small constant to prevent division by zero.
    
#     Returns:
#         None
#     """
#     base_params = base_model
#     lora_params = lora_model
    
#     # Track differences
#     param_diff = {}

#     for name, base_param in base_params.items():
#         if name in lora_params:
#             lora_param = lora_params[name]
#             # Check if shapes match
#             if base_param.shape != lora_param.shape:
#                 param_diff[name] = {
#                     "message": "Shape mismatch",
#                     "base_shape": base_param.shape,
#                     "lora_shape": lora_param.shape
#                 }
#                 continue
            
#             # Calculate the norm difference between parameters
#             norm_difference = torch.norm(base_param - lora_param).item()
#             base_norm = torch.norm(base_param).item()
            
#             # Calculate percentage difference
#             percentage_difference = (norm_difference / (base_norm + epsilon)) * 100
            
#             if norm_difference > 0:  # Record only significant differences
#                 print("\nParameters with differences:")
#                 for name, diff in param_diff.items():
#                     print(f"{name}: Absolute Difference = {diff['absolute_diff']:.6f}, Percentage Difference = {diff['percentage_diff']:.2f}%")
                
#                 param_diff[name] = {
#                     "absolute_diff": norm_difference,
#                     "percentage_diff": percentage_difference
#                 }
#         else:
#             param_diff[name] = {"message": "Parameter missing in the combined model"}
    
#     # Parameters present in LoRA but not in the base model
#     for name in lora_params.keys():
#         if name not in base_params:
#             param_diff[name] = {"message": "Parameter only in combined model"}

#     # Save the differences to a file
#     with open(output_file, "w") as f:
#         json.dump(param_diff, f, indent=4)
#     print(f"Differences saved to {output_file}")


def compare_model_parameters_with_percentage_to_file(base_model, lora_model, output_file, epsilon=1e-10):
    """
    Compare parameters between a base model and a combined model (base + LoRA),
    calculate percentage differences (rounded to 3 decimal places), and save to a single file.
    Includes the '%' sign in the output.
    
    Args:
        base_model (OrderedDict): State dict of the base model.
        lora_model (OrderedDict): State dict of the combined model.
        output_file (str): Path to save the results.
        epsilon (float): Small constant to prevent division by zero.
    
    Returns:
        None
    """
    base_params = base_model
    lora_params = lora_model
    
    # Dictionary to store differences
    param_diff = {}
    
    total_percentage_diff = 0
    total_params_count = 0

    for name, base_param in base_params.items():
        if name in lora_params:
            lora_param = lora_params[name]
            
            # Check if shapes match
            if base_param.shape != lora_param.shape:
                param_diff[name] = {
                    "message": "Shape mismatch",
                    "base_shape": base_param.shape,
                    "lora_shape": lora_param.shape
                }
                continue
            
            # Calculate percentage difference
            norm_difference = torch.norm(base_param - lora_param).item()
            base_norm = torch.norm(base_param).item()
            percentage_difference = round((norm_difference / (base_norm + epsilon)) * 100, 3)
            
            # Update totals for averages
            total_percentage_diff += percentage_difference
            total_params_count += 1
            
            # Store individual parameter differences
            param_diff[name] = {
                "percentage_diff": f"{percentage_difference}%"  # Add percentage sign
            }
        else:
            param_diff[name] = {"message": "Parameter missing in the combined model"}
    
    # Parameters present in LoRA but not in the base model
    for name in lora_params.keys():
        if name not in base_params:
            param_diff[name] = {"message": "Parameter only in combined model"}
    
    # Calculate average percentage difference
    if total_params_count > 0:
        average_percentage_diff = round(total_percentage_diff / total_params_count, 3)
    else:
        average_percentage_diff = 0.0
    
    # Add average percentage difference to the output
    param_diff["summary"] = {
        "average_percentage_diff": f"{average_percentage_diff}%"  # Add percentage sign
    }
    
    # Save all differences to a single file
    with open(output_file, "w") as f:
        json.dump(param_diff, f, indent=4)
    
    print(f"Differences saved to {output_file}")


original_base_model_weights = torch.load(base_model_weights_path, map_location=device)

compare_model_parameters_with_percentage_to_file(
    base_model=original_base_model_weights,
    lora_model=combined_model_with_filtered_general_qa_lora_weights,
    output_file=combined_general_qa_lora_param_difference_file_path
)
