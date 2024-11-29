#from huggingface_hub import notebook_login
#from google.colab import userdata
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
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
import os
import numpy as np
import transformers

hf_token = "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"

#root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/"
root_folder = "/home/ums/Desktop"
dataset_folder = os.path.join(root_folder, "add_multiple_lora_to_base_model", "dataset")
medical_qa_dataset_folder = os.path.join(dataset_folder, "medical_qa")

base_model_path = os.path.join(root_folder, "add_multiple_lora_to_base_model", "gemma_2b_base_model.pt")
medical_qa_lora_path = os.path.join(root_folder, "add_multiple_lora_to_base_model", "Medical QA Weights/medical_qa_lora_model.pt")
merged_peft_medical_qa_lora_path = os.path.join(root_folder, "add_multiple_lora_to_base_model", "Medical QA Weights/merged_peft_medical_qa_lora.pt")

########### medical QA json files
medical_qa_train_json_path = os.path.join(medical_qa_dataset_folder, "train.jsonl")
medical_qa_val_json_path = os.path.join(medical_qa_dataset_folder, "val.jsonl")


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Enter "huggingface-cli login" in the command prompt
Enter hf_token into the command prompt
Token has not been saved to git credential helper.
Your token has been saved to /home/johnathon/.cache/huggingface/token
"""

### define the parameters
model_name = "google/gemma-2b"
new_model = "gemma-ft"

# Load the entire model on the GPU 0
# device_map = {"": 0}
# "auto" for multple GPUs
device_map = "auto"
#device_map = {"": 0}

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,
    #quantization_config = bnb_config,
    device_map = device_map)
# ).to(device)

torch.save(base_model.state_dict(), base_model_path)

base_model.config.use_cache = False

target_modules = []
target_modules_list = list(set(get_specific_layer_names(base_model)))
for i in target_modules_list:
    if len(i) == 0:
        continue
    else:
        target_modules.append(i)

print(target_modules)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)

# end of sequence token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # fix weird overflow issue with fp16 training

def get_train_val_dataset(json_file_path):
    dataset = []
    with jl.open(json_file_path, 'r') as reader:
        for data in reader:
            # Get "instruction" and "response" fields directly
            instruction = data.get("instruction", "")
            response = data.get("response", "Unanswerable")  # Default to "Unanswerable" if no response

            # Format as "Instruction" and "Response" for the filtered dataset
            formatted_text = "Instruction:\n{}\n\nResponse:\n{}".format(instruction, response)
            dataset.append({"text": formatted_text})
    return dataset


medical_qa_train_dataset = get_train_val_dataset(medical_qa_train_json_path)
medical_qa_val_dataset = get_train_val_dataset(medical_qa_val_json_path)

# print(len(medical_qa_train_dataset))
# print(len(medical_qa_val_dataset))

# Convert filtered_dataset (a list of dictionaries) to a Hugging Face Dataset
medical_qa_train_dataset_custom = Dataset.from_list(medical_qa_train_dataset)
medical_qa_val_dataset_custom = Dataset.from_list(medical_qa_val_dataset)

#### Finetune using parameter efficient finetuning

### define the parameters
model_name = "google/gemma-2b"
new_model = "gemma-ft"

################################################################
# LoRA parameters
################################################################
# LoRA attention dimension
# also known as rank, higher number for gpu with higher memory space
# lora_r = 64
lora_r = 256

# Alpha parameter for LoRA scaling
lora_alpha = 512

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################
# bitsandbytes parameters
################################################################
# Activate 4-bit precision base model loading
use_4bit = True

# compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################
# Training Arguments Parameters
################################################################

medical_qa_train_total_samples = len(medical_qa_train_dataset_custom)

# output directory where the model predictions and checkpoints will be stored
output_dir = "./results/results_medical_qa"

# number of training epochs
#num_train_epochs = 1
num_train_epochs = 3

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 8

# Batch size per GPU for evaluation
per_device_eval_batch_size = 8

# Number of update steps to accumulate the gradients for
#gradient_accumulation_steps = 1
gradient_accumulation_steps = 8

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# optimizer to use 
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
# -1 means not using max_steps
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# save checkpoint every x updates steps
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps

# Calculate steps per epoch
#save_steps = 25
steps_per_epoch_medical_qa = medical_qa_train_total_samples // effective_batch_size

save_steps_medical_qa = steps_per_epoch_medical_qa // 5  # Saves 5 times per epoch

# log every X updates steps
#logging_steps = 25
logging_steps_medical_qa = steps_per_epoch_medical_qa // 5  # Logs 5 times per epoch

################################################################
# SFT Parameters
################################################################

# Maximum sequence length to use
# more for higher compute
max_seq_length = 512 # None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = True # False

# Load QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit, # Activates 4-bit precision loading
    bnb_4bit_quant_type = bnb_4bit_quant_type, # nf4
    bnb_4bit_compute_dtype = compute_dtype, # float16
    bnb_4bit_use_double_quant = use_nested_quant # False
)

# Check GPU compatibility with bfloat116
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("Setting BF16 to True")
        bf16 = True
    else:
        bf16 = False
        
######## Load LoRA configuration
peft_medical_qa_config = LoraConfig(
    lora_alpha = lora_alpha,
    #lora_dropout = lora_dropout,
    r=lora_r,
    bias = "none",
    task_type="CAUSAL_LM",
    # attention modules
    target_modules=['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj']
)


# Set training parameters
training_arguments_medical_qa = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps_medical_qa,
    logging_steps = logging_steps_medical_qa,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    fp16 = fp16,
    bf16 = bf16,
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    lr_scheduler_type = lr_scheduler_type,
    report_to = "tensorboard",
    load_best_model_at_end=True,
    #evaluation_strategy="steps",  # Evaluate at the end of each epoch
    #save_strategy="steps",        # Save at the end of each epoch
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save at the end of each epoch
    save_total_limit=1,   
)


trainer_medical_qa = SFTTrainer(
    model = base_model,
    train_dataset = medical_qa_train_dataset_custom,
    eval_dataset = medical_qa_val_dataset_custom,
    peft_config = peft_medical_qa_config,
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
    tokenizer = tokenizer,
    args = training_arguments_medical_qa, 
    packing = packing
)

######### save only the lora weights

def save_lora_adapter(trainer, file_path):
    """
    Save only the LoRA adapter weights from the model in FP16 precision.

    Args:
        trainer (SFTTrainer): The trainer containing the fine-tuned model.
        file_path (str): Path to save the LoRA adapter weights.
    """
    # Filter LoRA-specific weights
    lora_adapter = {
        k: v.to(torch.float16)  # Save in FP16 for reduced size
        for k, v in trainer.model.state_dict().items()
        if "lora" in k
    }

    # Save the filtered weights
    torch.save(lora_adapter, file_path)
    print(f"LoRA adapter successfully saved in FP16 to {file_path}")

### train medical qa
trainer_medical_qa.train(resume_from_checkpoint=None)

# save lora 
save_lora_adapter(trainer_medical_qa, medical_qa_lora_path)
trainer_medical_qa.model.save_pretrained(new_model)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
merged_model = fine_tuned_model.merge_and_unload()


# Save the merged model as a .pt file
torch.save(merged_model.state_dict(), merged_peft_medical_qa_lora_path)


# for the saved lora model and merged model
def check_for_lora_weights_in_dict(state_dict):
    # Look for layer names specific to LoRA (commonly "_lora" or similar identifiers)
    lora_layers = [name for name in state_dict.keys() if "_lora" in name or "lora" in name.lower()]

    if lora_layers:
        print("LoRA weights found in the model state_dict:")
        for name in lora_layers:
            print(f"{name}: {state_dict[name].shape} - Sum of values: {state_dict[name].sum().item()}")
    else:
        print("No LoRA weights found in the model state_dict.")


# Function to check for LoRA weights for base model
def check_for_lora_weights(model):
    # Look for layer names specific to LoRA (commonly "_lora" or similar identifiers)
    lora_layers = [name for name in model.state_dict().keys() if "_lora" in name or "lora" in name.lower()]

    if lora_layers:
        print("LoRA weights found in the model:")
        for name in lora_layers:
            print(f"{name}: {model.state_dict()[name].shape} - Sum of values: {model.state_dict()[name].sum().item()}")
    else:
        print("No LoRA weights found in the model.")

print("==================== Check if lora exist in based model ===========================")
base_model_weights = torch.load(base_model_path, map_location="cuda:3")
#Check for LoRA weights in the loaded model
check_for_lora_weights_in_dict(base_model_weights)

print("==================== Check if lora exist in merged based model ===========================")
merge_model_weights = torch.load(merged_peft_medical_qa_lora_path, map_location="cuda:3")
#Check for LoRA weights in the merge_model_weights model
check_for_lora_weights_in_dict(merge_model_weights)

print("==================== Check if lora exist in saved lora model only===========================")
lora_model_weights = torch.load(medical_qa_lora_path, map_location="cuda:3")
#Check for LoRA weights in the merge_model_weights model
check_for_lora_weights_in_dict(lora_model_weights)


