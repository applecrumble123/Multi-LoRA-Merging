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
medical_autoregressive_lora_path = os.path.join(root_code_folder, "Medical autoregressive weights/medical_autoregressive_lora.pt")

combined_model_with_general_qa_medical_autoregressive_lora_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/combined_model_with_general_qa_medical_autoregressive_lora.pt")

combined_general_qa_medical_autoregressive_lora_param_difference_file_path = os.path.join(root_code_folder, "General QA + Medical Autoregessive/general_qa_medical_autoregressive_lora_param_difference.txt")

# base_model_weights = torch.load(base_model_weights_path, map_location=device)
# general_qa_lora_weights = torch.load(general_qa_lora_path, map_location=device)
# medical_autoregressive_lora_weights = torch.load(medical_autoregressive_lora_path, map_location=device)

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
filtered_medical_autoregressive_lora_weights = torch.load(medical_autoregressive_lora_path, map_location=device)

# print("Keys in filtered_general_qa_lora_weights:")
# for key in filtered_general_qa_lora_weights.keys():
#     print(key)



target_modules = ['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj']

# def combine_2_lora_and_base(loaded_base_model_weights, filtered_lora_weights_1, filtered_lora_weights_2, save_path, 
#                             alpha_weight_1, rank_weight_1, alpha_weight_2, rank_weight_2, save:bool = True):
#     for name, base_weight in loaded_base_model_weights.items():
#         matched = False
#         for target in target_modules:
#             if f".{target}." in name:
#                 matched = True
#                 print(f"Processing {name} in target module '{target}'")
                
#                 # Construct the expected LoRA weight keys
#                 base_name_no_suffix = name.replace(".weight", "")
#                 lora_a_key_1 = f"base_model.model.{base_name_no_suffix}.lora_A.default.weight"
#                 lora_b_key_1 = f"base_model.model.{base_name_no_suffix}.lora_B.default.weight"
#                 lora_a_key_2 = f"base_model.model.{base_name_no_suffix}.lora_A.default.weight"
#                 lora_b_key_2 = f"base_model.model.{base_name_no_suffix}.lora_B.default.weight"
                
#                 # Check if LoRA keys exist in both models and log their presence
#                 if (lora_a_key_1 in filtered_lora_weights_1 and lora_b_key_1 in filtered_lora_weights_1 and 
#                     lora_a_key_2 in filtered_lora_weights_2 and lora_b_key_2 in filtered_lora_weights_2):
                    
#                     print(f"Found matching LoRA weights in both models: {lora_a_key_1} and {lora_b_key_1}")

#                     # Extract LoRA weights
#                     # Extract LoRA weights for both models
#                     lora_A_1 = filtered_lora_weights_1[lora_a_key_1]
#                     lora_B_1 = filtered_lora_weights_1[lora_b_key_1]
#                     lora_A_2 = filtered_lora_weights_2[lora_a_key_2]
#                     lora_B_2 = filtered_lora_weights_2[lora_b_key_2]
                    
#                     print("lora_A_1 shape: ", lora_A_1.shape)
#                     print("lora_B_1 shape: ", lora_B_1.shape)
#                     print("lora_A_2 shape: ", lora_A_2.shape)
#                     print("lora_B_2 shape: ", lora_B_2.shape)

#                     # Compute the low-rank updates for each LoRA model
#                     delta_weight_1 = lora_B_1 @ lora_A_1
#                     delta_weight_2 = lora_B_2 @ lora_A_2
#                     print(f"Computed delta_weight_1 factor with shape: {delta_weight_1.shape}")
#                     print(f"Computed delta_weight_2 factor with shape: {delta_weight_2.shape}")

#                     # Scale the base weight by the computed scaling factor
#                     # Ensure the shapes match for element-wise multiplication
#                     if delta_weight_1.shape == base_weight.shape and delta_weight_2.shape == base_weight.shape:
#                         # Apply the scaling formula: w = w + w * (A * B^T)
#                         #base_model_weights_new[name] = base_weight + base_weight * delta_weight
#                         loaded_base_model_weights[name] = loaded_base_model_weights[name] + (alpha_weight_1/rank_weight_1) * delta_weight_1 
#                         loaded_base_model_weights[name] = loaded_base_model_weights[name] + (alpha_weight_2/rank_weight_2) * delta_weight_2
#                         #loaded_base_model_weights[name] = loaded_base_model_weights[name] + delta_weight_2
#                         #loaded_base_model_weights[name] = base_weight + 2 * delta_weight_1 +  2* delta_weight_2
#                         #base_model_weights_new[name] = base_weight + base_weight * scaling_factor
#                     else:
#                         #base_model_weights_new[name] = base_weight
#                         print(f"Shape mismatch for {name}: base weight shape {base_weight.shape} vs scaling factor shape {delta_weight_1.shape}")
#                         print(f"Shape mismatch for {name}: base weight shape {base_weight.shape} vs scaling factor shape {delta_weight_2.shape}")
#                 else:
#                     print(f"LoRA weights not found for {name}. Skipping update.")
#                 break
        
#         # Ensure skipping message prints only once if no match
#         if not matched:
#             print(f"Skipping {name}, not in target modules.")

#     # Now, create a new state_dict for saving
#     # updated_state_dict = base_model_weights.copy()
#     # updated_state_dict.update(base_model_weights_new)
#     if save == True:
#         torch.save(loaded_base_model_weights, save_path)

def combine_2_lora_and_base(loaded_base_model_weights, filtered_lora_weights_1, filtered_lora_weights_2, save_path, 
                            alpha_weight_1, rank_weight_1, alpha_weight_2, rank_weight_2, save: bool = True):
    for name, base_weight in loaded_base_model_weights.items():
        matched = False
        for target in target_modules:
            if f".{target}." in name:
                matched = True
                print(f"Processing {name} in target module '{target}'")
                
                # Construct the expected LoRA weight keys
                base_name_no_suffix = name.replace(".weight", "")
                lora_a_key_1 = f"base_model.model.{base_name_no_suffix}.lora_A.default.weight"
                lora_b_key_1 = f"base_model.model.{base_name_no_suffix}.lora_B.default.weight"
                lora_a_key_2 = f"base_model.model.{base_name_no_suffix}.lora_A.default.weight"
                lora_b_key_2 = f"base_model.model.{base_name_no_suffix}.lora_B.default.weight"
                
                # Check if LoRA keys exist in both models
                if (lora_a_key_1 in filtered_lora_weights_1 and lora_b_key_1 in filtered_lora_weights_1 and 
                    lora_a_key_2 in filtered_lora_weights_2 and lora_b_key_2 in filtered_lora_weights_2):
                    
                    print(f"Found matching LoRA weights in both models: {lora_a_key_1} and {lora_b_key_1}")

                    # Extract LoRA weights
                    lora_A_1 = filtered_lora_weights_1[lora_a_key_1]
                    lora_B_1 = filtered_lora_weights_1[lora_b_key_1]
                    lora_A_2 = filtered_lora_weights_2[lora_a_key_2]
                    lora_B_2 = filtered_lora_weights_2[lora_b_key_2]

                    # Compute the low-rank updates for each LoRA model
                    delta_weight_1 = lora_B_1 @ lora_A_1
                    delta_weight_2 = lora_B_2 @ lora_A_2

                    delta_norm_1 = torch.norm(delta_weight_1)
                    delta_norm_2 = torch.norm(delta_weight_2)

                    delta_weight_1_normalized = delta_weight_1 / delta_norm_1
                    delta_weight_2_normalized = delta_weight_2 / delta_norm_2

                    print(f"Norm of delta_weight_1: {delta_norm_1.item()}, Norm of delta_weight_2: {delta_norm_2.item()}")

                    # Combine the updates with normalization
                    total_weight = (alpha_weight_1 / rank_weight_1) + (alpha_weight_2 / rank_weight_2)

                    lambda_1 = delta_norm_2 / (delta_norm_1 + delta_norm_2)
                    lambda_2 = delta_norm_1 / (delta_norm_1 + delta_norm_2)
                    # combined_delta_weight = ((alpha_weight_1 / rank_weight_1) * delta_weight_1 +
                    #                          (alpha_weight_2 / rank_weight_2) * delta_weight_2) / total_weight

                    # # Weighted Contribution Based on Norm
                    # # Balance the updates inversely proportional to their norms
                    # combined_delta_weight = ((delta_norm_2 / (delta_norm_1 + delta_norm_2)) * delta_weight_1 +
                    #      (delta_norm_1 / (delta_norm_1 + delta_norm_2)) * delta_weight_2)

                    if delta_weight_1.shape == base_weight.shape:
                        # loaded_base_model_weights[name] += combined_delta_weight
                        loaded_base_model_weights[name] =  loaded_base_model_weights[name] +  0.7 *  delta_weight_1 / 2
                        loaded_base_model_weights[name] =  loaded_base_model_weights[name] +    delta_weight_2 / 2

                        #loaded_base_model_weights[name] =  loaded_base_model_weights[name] +   delta_weight_2
                        #loaded_base_model_weights[name] =  loaded_base_model_weights[name] +  lambda_2 * delta_weight_2 / total_weight
                        
                    else:
                        print(f"Shape mismatch for {name}: base weight shape {base_weight.shape} vs delta weight shape {combined_delta_weight.shape}")
                else:
                    print(f"LoRA weights not found for {name}. Skipping update.")
                break
        
        if not matched:
            print(f"Skipping {name}, not in target modules.")

    if save:
        torch.save(loaded_base_model_weights, save_path)


base_model_weights_general_qa_medical_autoregressive = torch.load(base_model_weights_path, map_location=device)

combine_2_lora_and_base(loaded_base_model_weights = base_model_weights_general_qa_medical_autoregressive, 
                        filtered_lora_weights_1 = filtered_general_qa_lora_weights, 
                        filtered_lora_weights_2 = filtered_medical_autoregressive_lora_weights,
                        save_path = combined_model_with_general_qa_medical_autoregressive_lora_path,
                        alpha_weight_1 = 256, 
                        rank_weight_1 = 128,
                        alpha_weight_2 = 512, 
                        rank_weight_2 = 256,
                        save = True)

combined_model_with_filtered_general_qa_general_qa_medical_autoregressive_lora_weights = torch.load(combined_model_with_general_qa_medical_autoregressive_lora_path, map_location=device)

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
    lora_model=combined_model_with_filtered_general_qa_general_qa_medical_autoregressive_lora_weights,
    output_file=combined_general_qa_medical_autoregressive_lora_param_difference_file_path
)
