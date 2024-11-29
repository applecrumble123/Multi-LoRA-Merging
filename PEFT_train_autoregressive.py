#from huggingface_hub import notebook_login
#from google.colab import userdata
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, TrainerCallback
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from collections import defaultdict
import jsonlines as jl
from peft import LoraConfig, TaskType, PeftModel
#from peft import LoraConfig, TaskType
#from new_peft.src.peft import PeftModel, LoraConfig, TaskType
import os
from datasets import Dataset
from torch.utils.data import DataLoader
import transformers
import os

# Limit the visible devices to GPU 2 and GPU 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

hf_token = "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"
#notebook_login()

# Load the entire model on the GPU 0
# device_map = {"": 0}
# "auto" for multple GPUs
device_map = "auto"



# Also, ensure all data tensors and operations use only these GPUs
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

"""
Enter "huggingface-cli login" in the command prompt
Enter hf_token into the command prompt
Token has not been saved to git credential helper.
Your token has been saved to /home/johnathon/.cache/huggingface/token
"""

#root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop"
root_folder = "/home/ums/Desktop"
root_code_folder = os.path.join(root_folder, "add_multiple_lora_to_base_model")
dataset_folder = os.path.join(root_code_folder, "dataset")
medical_documents_folder = os.path.join(dataset_folder, "medical_documents/train")
medical_regressive_lora_path = os.path.join(root_code_folder, "Medical autoregressive weights/medical_autoregressive_lora.pt")

merged_peft_autogressive_medical_lora_path =  os.path.join(root_folder, "add_multiple_lora_to_base_model", "Medical autoregressive weights/merged_peft_medical_autoregressive_lora.pt")


### define the parameters
model_name = "google/gemma-2b"
new_model = "gemma-ft"
######## Load the model and tokenizer
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

#### dataset

# Read all text files into a list of documents
documents = []
for filename in os.listdir(medical_documents_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(medical_documents_folder, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())

window_size = 512

# Adjust window size and causal mask logic
window_size = min(window_size, base_model.config.max_position_embeddings)



# Function to tokenize with sliding window
def tokenize_with_window(text, window_size):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=window_size,
        padding="max_length",
    )
    return {
        "input_ids": tokens["input_ids"].squeeze(0),  # Remove extra dimensions if necessary
        "attention_mask": tokens["attention_mask"].squeeze(0),
    }


# Create a dataset from documents with sliding windows
input_ids, attention_masks = [], []
for doc in documents:
    for i in range(0, len(doc), window_size):
        window = doc[i : i + window_size]
        tokens = tokenize_with_window(window, window_size)
        input_ids.append(tokens["input_ids"])
        attention_masks.append(tokens["attention_mask"])

# Convert lists of tensors to a single stacked tensor
input_ids = torch.stack(input_ids)  # Shape: [num_windows, window_size]
attention_masks = torch.stack(attention_masks)  # Shape: [num_windows, window_size]

# Clone the `input_ids` for labels and shift them
labels = input_ids.clone()
labels = labels[:, 1:].contiguous()  # Shift labels to the left by 1
input_ids = input_ids[:, :-1].contiguous()  # Remove the last token in `input_ids`

# Mask padding tokens in the labels with -100 so they are ignored in the loss calculation
labels[labels == tokenizer.pad_token_id] = -100

# Update dataset to include labels
dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": attention_masks,
    "labels": labels  # The shifted input sequence as target labels
})

# DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


################################################################
# LoRA parameters
################################################################
# LoRA attention dimension
# also known as rank, higher number for gpu with higher memory space
# lora_r = 64
# lora_r = 64
lora_r = 256

# Alpha parameter for LoRA scaling
lora_alpha = 512

# Dropout probability for LoRA layers
#lora_dropout = 0.1
lora_dropout = 0.01

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

# output directory where the model predictions and checkpoints will be stored
output_dir = "./results/results_medical_autoregressive"

# number of training epochs
num_train_epochs = 3

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 8

# Batch size per GPU for evaluation
per_device_eval_batch_size = 8

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 8

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
#learning_rate = 1e-5
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
save_steps = 400

# log every X updates steps
logging_steps = 400

################################################################
# SFT Parameters
################################################################

# Maximum sequence length to use
# more for higher compute
#max_seq_length = 40 # None
max_seq_length = 512

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
        

# Define a custom callback to track and save the lowest training loss
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

# callback = SaveLowestTrainingLossCallback(medical_regressive_lora_path)

######## Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    #lora_dropout = lora_dropout,
    r=lora_r,
    bias = "none",
    task_type="CAUSAL_LM",
    # attention modules
    target_modules=['v_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj', 'o_proj', 'k_proj']
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
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
    save_strategy="no",        # Save at the end of each epoch
)

print(training_arguments)


model = PeftModel(base_model, peft_config)
model.train()


trainer_autoregresssive = Trainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = training_arguments,
    #callbacks=[callback] 
)

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
trainer_autoregresssive.train(resume_from_checkpoint=None)

# save lora 
save_lora_adapter(trainer_autoregresssive, medical_regressive_lora_path)

trainer_autoregresssive.model.save_pretrained(new_model)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
merged_model = fine_tuned_model.merge_and_unload()


# Save the merged model as a .pt file
torch.save(merged_model.state_dict(), merged_peft_autogressive_medical_lora_path)


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


print("==================== Check if lora exist in merged based model ===========================")
merge_model_weights = torch.load(merged_peft_autogressive_medical_lora_path, map_location="cuda:3")
#Check for LoRA weights in the merge_model_weights model
check_for_lora_weights_in_dict(merge_model_weights)

print("==================== Check if lora exist in saved lora model only===========================")
lora_model_weights = torch.load(medical_regressive_lora_path, map_location="cuda:3")
#Check for LoRA weights in the merge_model_weights model
check_for_lora_weights_in_dict(lora_model_weights)