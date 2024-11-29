import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluate import load
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
model_name = "google/gemma-2b"
hf_token = "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"

# Paths
root_folder = "/home/ums/Desktop"
root_code_folder = os.path.join(root_folder, "add_multiple_lora_to_base_model")

# Dynamically populate the list of test document paths
test_documents_folder = os.path.join(root_code_folder, "dataset/medical_documents/test")
test_documents_list = [
    os.path.join(test_documents_folder, file)
    for file in os.listdir(test_documents_folder)
    if os.path.isfile(os.path.join(test_documents_folder, file)) and file.endswith(".txt")
]

# Print the dynamically created list to verify
print("Test Documents List:")
for path in test_documents_list:
    print(path)


base_model_weights_path = os.path.join(root_code_folder, "gemma_2b_base_model.pt")
combined_weights_path = os.path.join(root_code_folder, "Medical autoregressive weights/combined_model_with_filtered_medical_autoregressive_lora.pt")
merged_peft_weights_path = os.path.join(root_code_folder, "Medical autoregressive weights/merged_peft_medical_autoregressive_lora.pt")

output_file_path = os.path.join(root_code_folder, "Medical autoregressive weights/autoregressive_answers_combined.txt")

# Function to chunk text
def chunk_text(text, window_size=512, max_chunks=100):
    return [text[i: i + window_size] for i in range(0, len(text), window_size)][:max_chunks]

# Function to calculate averages and standard deviations of ROUGE scores
def calculate_avg_std(rouge_scores):
    base_scores = [score["base"] for score in rouge_scores]
    combined_scores = [score["combined"] for score in rouge_scores]
    merged_peft_scores = [score["merged_peft"] for score in rouge_scores]
    return {
        "base_avg": np.mean(base_scores),
        "base_std": np.std(base_scores),
        "combined_avg": np.mean(combined_scores),
        "combined_std": np.std(combined_scores),
        "merged_peft_avg": np.mean(merged_peft_scores),
        "merged_peft_std": np.std(merged_peft_scores),
    }

# Function to save all answers into a single file
def save_all_answers_to_single_file(output_file_path, answers):
    with open(output_file_path, "w", encoding="utf-8") as file:
        for document_idx, document_answers in enumerate(answers):
            file.write(f"==== Document {document_idx + 1} ====\n\n")
            for i, (ref, base, combined, merged_peft) in enumerate(zip(
                document_answers["chunks"], 
                document_answers["base"], 
                document_answers["combined"], 
                document_answers["merged_peft"]
            )):
                file.write(f"Reference Answer {i+1}:\n{ref}\n\n")
                file.write(f"Generated Answer (Base Model) {i+1}:\n{base}\n\n")
                file.write(f"Generated Answer (Combined Model) {i+1}:\n{combined}\n\n")
                file.write(f"Generated Answer (Merged PEFT Model) {i+1}:\n{merged_peft}\n\n")
                file.write("=" * 80 + "\n\n")
    print(f"Reference and generated answers saved to {output_file_path}")

# Load models and weights
def load_models_and_weights():
    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)
    combined_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)
    merged_peft_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float16)

    base_model.load_state_dict(torch.load(base_model_weights_path, map_location="cuda"))
    combined_model.load_state_dict(torch.load(combined_weights_path, map_location="cuda"))
    merged_peft_model.load_state_dict(torch.load(merged_peft_weights_path, map_location="cuda"))

    base_model.to(device)
    combined_model.to(device)
    merged_peft_model.to(device)

    return base_model, combined_model, merged_peft_model

# Main execution
if __name__ == "__main__":
    # Load models
    base_model, combined_model, merged_peft_model = load_models_and_weights()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Initialize ROUGE metric
    rouge = load("rouge")

    # Store all results
    all_document_answers = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    # Iterate through test documents
    for document_path in test_documents_list:
        with open(document_path, "r", encoding="utf-8") as file:
            test_document = file.read()

        test_chunks = chunk_text(test_document)

        # Store generated answers for this document
        base_generated_answers = []
        combined_generated_answers = []
        merged_peft_generated_answers = []

        # Evaluate models on each chunk
        for chunk in test_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                # Base model predictions
                base_output = base_model.generate(inputs["input_ids"], max_length=512)
                base_prediction = tokenizer.decode(base_output[0], skip_special_tokens=True)
                base_generated_answers.append(base_prediction)

                # Combined model predictions
                combined_output = combined_model.generate(inputs["input_ids"], max_length=512)
                combined_prediction = tokenizer.decode(combined_output[0], skip_special_tokens=True)
                combined_generated_answers.append(combined_prediction)

                # Merged PEFT model predictions
                merged_peft_output = merged_peft_model.generate(inputs["input_ids"], max_length=512)
                merged_peft_prediction = tokenizer.decode(merged_peft_output[0], skip_special_tokens=True)
                merged_peft_generated_answers.append(merged_peft_prediction)

            # Compute ROUGE scores
            base_rouge_score = rouge.compute(predictions=[base_prediction], references=[chunk])
            combined_rouge_score = rouge.compute(predictions=[combined_prediction], references=[chunk])
            merged_peft_rouge_score = rouge.compute(predictions=[merged_peft_prediction], references=[chunk])

            rouge1_scores.append({
                "base": base_rouge_score["rouge1"],
                "combined": combined_rouge_score["rouge1"],
                "merged_peft": merged_peft_rouge_score["rouge1"],
            })
            rouge2_scores.append({
                "base": base_rouge_score["rouge2"],
                "combined": combined_rouge_score["rouge2"],
                "merged_peft": merged_peft_rouge_score["rouge2"],
            })
            rougeL_scores.append({
                "base": base_rouge_score["rougeL"],
                "combined": combined_rouge_score["rougeL"],
                "merged_peft": merged_peft_rouge_score["rougeL"],
            })

        # Save answers for this document
        all_document_answers.append({
            "chunks": test_chunks,
            "base": base_generated_answers,
            "combined": combined_generated_answers,
            "merged_peft": merged_peft_generated_answers,
        })

    # Save results to file
    save_all_answers_to_single_file(output_file_path, all_document_answers)

    # Calculate averages
    avg_rouge1 = calculate_avg_std(rouge1_scores)
    avg_rouge2 = calculate_avg_std(rouge2_scores)
    avg_rougeL = calculate_avg_std(rougeL_scores)

    # Print averages
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1 Average (Base): {avg_rouge1['base_avg']:.4f}, Std Dev: {avg_rouge1['base_std']:.4f}")
    print(f"ROUGE-1 Average (Combined): {avg_rouge1['combined_avg']:.4f}, Std Dev: {avg_rouge1['combined_std']:.4f}")
    print(f"ROUGE-1 Average (Merged PEFT): {avg_rouge1['merged_peft_avg']:.4f}, Std Dev: {avg_rouge1['merged_peft_std']:.4f}")

    print(f"ROUGE-2 Average (Base): {avg_rouge2['base_avg']:.4f}, Std Dev: {avg_rouge2['base_std']:.4f}")
    print(f"ROUGE-2 Average (Combined): {avg_rouge2['combined_avg']:.4f}, Std Dev: {avg_rouge2['combined_std']:.4f}")
    print(f"ROUGE-2 Average (Merged PEFT): {avg_rouge2['merged_peft_avg']:.4f}, Std Dev: {avg_rouge2['merged_peft_std']:.4f}")

    print(f"ROUGE-L Average (Base): {avg_rougeL['base_avg']:.4f}, Std Dev: {avg_rougeL['base_std']:.4f}")
    print(f"ROUGE-L Average (Combined): {avg_rougeL['combined_avg']:.4f}, Std Dev: {avg_rougeL['combined_std']:.4f}")
    print(f"ROUGE-L Average (Merged PEFT): {avg_rougeL['merged_peft_avg']:.4f}, Std Dev: {avg_rougeL['merged_peft_std']:.4f}")
