#from huggingface_hub import notebook_login
#from google.colab import userdata
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, TrainerCallback
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
general_qa_dataset_folder = os.path.join(dataset_folder, "general_qa")


base_model_path = os.path.join(root_folder, "add_multiple_lora_to_base_model", "gemma_2b_base_model.pt")
general_qa_lora_path = os.path.join(root_folder, "add_multiple_lora_to_base_model", "General QA Weights/general_qa_lora.pt")
merged_general_qa_lora_path = os.path.join(root_folder, "add_multiple_lora_to_base_model", "General QA Weights/merged_peft_general_qa_lora.pt")

########### general QA json files
general_qa_train_json_path = os.path.join(general_qa_dataset_folder, "general_qa_train_data.jsonl")
general_qa_val_json_path = os.path.join(general_qa_dataset_folder, "general_qa_val_data.jsonl")


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_token = "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"
#notebook_login()

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

"""
Enter "huggingface-cli login" in the command prompt
Enter hf_token into the command prompt
Token has not been saved to git credential helper.
Your token has been saved to /home/johnathon/.cache/huggingface/token
"""

# Initialize the list to store formatted entries
# general_qa_train_dataset = []
# general_qa_val_dataset = []

# medical_qa_train_dataset = []
# medical_qa_val_dataset = []

# with jl.open(train_save_json_file_path, 'r') as reader:
#     for data in reader:
#         # Extract "Instruction" and "Response" from the "text" field
#         text = data.get("text", "")
        
#         # Parse "Instruction" and "Response" from the text
#         if text:
#             parts = text.split("\n\nResponse:\n")
#             if len(parts) == 2:
#                 instruction = parts[0].replace("Instruction:\n", "").strip()
#                 response = parts[1].strip()
                
#                 # Format as "Instruction" and "Response" for the filtered dataset
#                 formatted_text = "Instruction:\n{}\n\nResponse:\n{}".format(instruction, response)
#                 filtered_dataset.append({"text": formatted_text})


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

general_qa_train_dataset = get_train_val_dataset(general_qa_train_json_path)
general_qa_val_dataset = get_train_val_dataset(general_qa_val_json_path)

# Load a sample from the dataset
# print(general_qa_train_dataset[0])

# Calculate the tokenized lengths of each example in the sample
general_qa_lengths = [len(tokenizer.encode(data["text"])) for data in general_qa_train_dataset]

# Analyze length statistics
general_qa_median_length = np.median(general_qa_lengths)
general_qa_max_length = np.max(general_qa_lengths)
general_qa_percentile_90 = np.percentile(general_qa_lengths, 90)

print(f"General QA Median token seq length: {general_qa_median_length}")
print(f"General QA  90th percentile token seq length: {general_qa_percentile_90}")
print(f"General QA  Maximum token seq length: {general_qa_max_length}")


# print(len(general_qa_train_dataset))
# print(len(general_qa_val_dataset))


# Convert filtered_dataset (a list of dictionaries) to a Hugging Face Dataset
general_qa_train_dataset_custom = Dataset.from_list(general_qa_train_dataset)
general_qa_val_dataset_custom = Dataset.from_list(general_qa_val_dataset)

#### Finetune using parameter efficient finetuning



################################################################
# LoRA parameters
################################################################
# LoRA attention dimension
# also known as rank, higher number for gpu with higher memory space
lora_r = 128
#lora_r = 4

# Alpha parameter for LoRA scaling
lora_alpha = 256

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

###############################################################
# Training Arguments Parameters
###############################################################

general_qa_train_total_samples = len(general_qa_train_dataset_custom)

# Output directory for model predictions and checkpoints
output_dir = "./results/results_general_qa"

# Number of training epochs (adjusted to balance overfitting and learning depth)
num_train_epochs_general_qa = 3
#num_train_epochs_general_qa = 1

# Enable bf16 for A100 GPUs
fp16 = False
bf16 = False  # Set bf16 to True for A100, maximizing memory efficiency

# Batch size per GPU for training and evaluation
per_device_train_batch_size =8
per_device_eval_batch_size = 8

# Number of gradient accumulation steps to achieve effective batch size
gradient_accumulation_steps = 8

# Enable gradient checkpointing for memory efficiency
gradient_checkpointing = True

# Gradient clipping to stabilize training
max_grad_norm = 0.3

# Learning rate (slightly lower for stability, particularly with medical QA)
learning_rate_general_qa = 2e-4
#learning_rate_general_qa = 2e-5
#learning_rate_general_qa = 8e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
#weight_decay = 0.01

# Optimizer and learning rate schedule
optim = "paged_adamw_32bit"
#optim =  "adamw_torch"
#lr_scheduler_type = "constant"
lr_scheduler_type = "cosine"

# Max steps, unused (-1) when training with epochs
max_steps = -1

# Warmup steps ratio
warmup_ratio = 0.03
#warmup_ratio = 0.1

# Group sequences by length for efficient memory usage
group_by_length = True

# Effective batch size calculation
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps

# Steps per epoch calculations for checkpoint saving and logging
steps_per_epoch_general_qa = general_qa_train_total_samples // effective_batch_size

# Checkpoint save frequency (save 5 times per epoch)
save_steps_general_qa = steps_per_epoch_general_qa // 5

# Logging frequency (log 5 times per epoch)
logging_steps_general_qa = steps_per_epoch_general_qa // 5


################################################################
# SFT Parameters
################################################################

# Maximum sequence length to use
# more for higher compute
max_seq_length = 128 # None

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
peft_config_general_qa = LoraConfig(
    lora_alpha = lora_alpha,
    #lora_dropout = 0.1,
    r=lora_r,
    bias = "none",
    task_type="CAUSAL_LM",
    # attention modules
    target_modules=['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj']
    )

# peft_config_general_qa = LoraConfig(
#     lora_alpha = 16,
#     #lora_dropout = 0.1,
#     r=32,
#     bias = "none",
#     task_type="CAUSAL_LM",
#     # attention modules
#     target_modules=['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj']
#     )


# Add the callback to the training arguments
# Create and add the callback for each trainer
# general_qa_callback = SaveLowestValidationLossLoraCallback(output_dir=output_dir, lora_save_path=general_qa_lora_path)
# medical_qa_callback = SaveLowestValidationLossLoraCallback(output_dir=output_dir, lora_save_path=medical_qa_lora_path)

# class SaveLowestTrainingLossCallback(TrainerCallback):
#     def __init__(self, model_save_path):
#         super().__init__()
#         self.best_training_loss = float("inf")  # Initialize with infinity
#         self.model_save_path = model_save_path

#     def on_log(self, args, state, control, **kwargs):
#         # Check if 'loss' is in logs to access the training loss
#         logs = kwargs.get("logs", {})
#         training_loss = logs.get("loss")

#         if training_loss is not None and training_loss < self.best_training_loss:
#             self.best_training_loss = training_loss
#             print(f"New best training loss: {self.best_training_loss:.4f}. Saving model...")
#             model = kwargs.get("model", None)
#             # Save only the model's LoRA layers for efficient storage
#             lora_adapter = {
#                 k: v.to(torch.float16)  # Save in FP16 for reduced size
#                 for k, v in kwargs['model'].state_dict().items()
#                 if "lora" in k
#             }

#             torch.save(lora_adapter, self.model_save_path)
#             print(f"LoRA adapter saved with lowest training loss at {self.model_save_path}")

# callback = SaveLowestTrainingLossCallback(general_qa_lora_path)

# Set training parameters
training_arguments_general_qa = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs_general_qa ,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps_general_qa,
    logging_steps = logging_steps_general_qa,
    learning_rate = learning_rate_general_qa ,
    weight_decay = weight_decay,
    fp16 = fp16,
    bf16 = bf16,
    # fp16 = False,
    # bf16 = False,
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    lr_scheduler_type = lr_scheduler_type,
    report_to = "tensorboard",
    load_best_model_at_end=True,
    #evaluation_strategy="steps",  # Evaluate at the end of each epoch
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save at the end of each epoch
    #save_strategy="no",        # Save at the end of each epoch
    save_total_limit=1,
    #overwrite_output_dir=True   
)



# Set supervised fine-tuning parameters
trainer_general_qa = SFTTrainer(
    model = base_model,
    train_dataset = general_qa_train_dataset_custom,
    eval_dataset = general_qa_val_dataset_custom,
    peft_config = peft_config_general_qa,
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
    tokenizer = tokenizer,
    args = training_arguments_general_qa, 
    packing = packing,
    #callbacks=[callback]
)



######### save only the lora weights
# def save_lora_adapter(trainer, file_path):
#     # Filter LoRA adapter layers
#     lora_adapter = {
#         k: v for k, v in trainer.model.state_dict().items() 
#         if "lora" in k or any(module in k for module in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])
#     }
#     torch.save(lora_adapter, file_path)
#     print(f"LoRA adapter saved to {file_path}")



# def save_lora_adapter(trainer, file_path):
#     # Filter LoRA adapter layers
#     lora_adapter = {
#         k: v.to(torch.float16) for k, v in trainer.model.state_dict().items() 
#         if "lora" in k or any(module in k for module in ['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj'])
#     }
#     torch.save(lora_adapter, file_path)
#     print(f"LoRA adapter saved in FP16 to {file_path}")

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

### train general qa
trainer_general_qa.train(resume_from_checkpoint=None)

save_lora_adapter(trainer_general_qa, general_qa_lora_path)

trainer_general_qa.model.save_pretrained(new_model)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
merged_model = fine_tuned_model.merge_and_unload()

#merged_model = trainer_general_qa.model.merge_and_unload()  # This merges LoRA layers into the base model

# Save the merged model as a .pt file
torch.save(merged_model.state_dict(), merged_general_qa_lora_path)

#save_lora_adapter(trainer_general_qa, general_qa_lora_path)
# save_lora_adapter(trainer_general_qa, general_qa_lora_path)

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
merge_model_weights = torch.load(merged_general_qa_lora_path, map_location="cuda:3")
#Check for LoRA weights in the merge_model_weights model
check_for_lora_weights_in_dict(merge_model_weights)

print("==================== Check if lora exist in saved lora model only===========================")
lora_model_weights = torch.load(general_qa_lora_path, map_location="cuda:3")
#Check for LoRA weights in the merge_model_weights model
check_for_lora_weights_in_dict(lora_model_weights)

######### uncomment to train the lora model
# train the model
#trainer.train()
# trainer.model.save_pretrained(new_model)

# save the entire model
# torch.save(trainer.model.state_dict(), "/mnt/c/Users/tohji/OneDrive/Desktop/lora.pt")


######### uncomment to save only the lora weights
# def save_lora_adapter(trainer, file_path):
#     # Filter LoRA adapter layers
#     lora_adapter = {
#         k: v for k, v in trainer.model.state_dict().items() 
#         if "lora" in k or any(module in k for module in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])
#     }
#     torch.save(lora_adapter, file_path)
#     print(f"LoRA adapter saved to {file_path}")

# trainer.train()
# # Save only LoRA adapter weights
# save_lora_adapter(trainer, "/mnt/c/Users/tohji/OneDrive/Desktop/single_lora.pt")

# trainer_part_1.train()
# save_lora_adapter(trainer_part_1, "/mnt/c/Users/tohji/OneDrive/Desktop/lora_1.pt")

# trainer_part_2.train()
# save_lora_adapter(trainer_part_2, "/mnt/c/Users/tohji/OneDrive/Desktop/lora_2.pt")

################ Prompt the newly fine-tuned model
# Run inference with the same prompt we used to test the pre-trained model

# same prompt used to test the pretrained model
#input_text = "What should I do on a trip to Europe"

# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     low_cpu_mem_usage = True,
#     return_dict = True,
#     torch_dtype=torch.float16,
#     device_map = device_map
# )




# uncomment to save the base model
# torch.save(base_model.state_dict(), "/mnt/c/Users/tohji/OneDrive/Desktop/gemma_2b_base.pt")

# load the base model
# base_model_weights = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/gemma_2b_base.pt", map_location="cuda")
# base_model.load_state_dict(base_model_weights, strict=False)

# Function to check for LoRA weights
def check_for_lora_weights(model):
    # Look for layer names specific to LoRA (commonly "_lora" or similar identifiers)
    lora_layers = [name for name in model.state_dict().keys() if "_lora" in name or "lora" in name.lower()]

    if lora_layers:
        print("LoRA weights found in the model:")
        for name in lora_layers:
            print(f"{name}: {model.state_dict()[name].shape} - Sum of values: {model.state_dict()[name].sum().item()}")
    else:
        print("No LoRA weights found in the model.")


# Check for LoRA weights in the loaded model
# check_for_lora_weights(base_model)

# load and merge the LoRA weights with the model weights
# only finetuned the LoRA layers --> need to merge base model and the adapter layers that are finetuned
# merge multple models?? or extend the PEFT class to merge multiple lora
# 2 domains 2 task --> 4 LoRa (finance, healthcare? task specific - QA, train-of-thoughts) --> 4 datasets
# model = PeftModel.from_pretrained(base_model, [new_model, new_model])
# model = model.merge_and_unload()

# single_lora_model  = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/single_lora.pt")

# lora_model_1  = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/lora_1.pt")
# lora_model_2  = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/lora_2.pt")


