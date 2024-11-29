# Multi-LoRA-Merging
## About
This study explores whether combining two Low-Rank Adaptation (LoRA) models, trained on distinct tasks and domains, can achieve results comparable or superior to a single task- and domain-specific LoRA. Specifically, we examine the potential of merging a general question-answering LoRA model and a medical-trained autoregressive LoRA model to improve performance in a medical question-answering task. Additionally, we evaluate a third configuration where both domain and task align with a medical QA LoRA, demonstrating domain-specific adaptability. The primary objective is to determine if such a combination eliminates the need to train task-specific LoRAs for each domain, thereby increasing efficiency. We trained individual LoRA models on the SQuADv2 dataset, a medical QA dataset, and a medical autoregressive task using domain-specific documents. Subsequently, we manually merged the LoRA weights with the base model and compared the results with the PEFT-merged model to validate equivalence. Furthermore, we evaluated the combined general and domain-specific LoRA models to assess their performance against single task/domain-specific LoRAs

LoRA Paper: https://arxiv.org/abs/2106.09685

Combining Two LoRA Models
To evaluate the feasibility of integrating two LoRA modules, we extended the merging process to incorporate updates from both models sequentially:
W' = W + (α1 / r1) ⋅ (B1 ⋅ A1) + (α2 / r2) ⋅ (B2 ⋅ A2)
where α1, r1 and α2, r2 are the scaling factors and ranks for the first and second LoRA models, respectively. The base model weights (W) are updated incrementally with the contributions of each LoRA module in a stepwise manner. The formula first applies the update from the first LoRA module (B1 ⋅ A1) scaled by (α1 / r1), modifying the base model weights to produce an intermediate state. Subsequently, the second module's update, (B2 ⋅ A2) scaled by (α2 / r2), is applied to this intermediate state, further refining the weights. Each step builds upon the results of the previous one.


## Dataset
### SQuADv2 Dataset
The Stanford Question Answering Dataset (SQuADv2) is a widely used benchmark for evaluating question-answering systems. It comprises over 150,000 questions paired with context paragraphs extracted from Wikipedia, covering a broad range of topics. A key distinction from its predecessor, SQuADv1, is the inclusion of unanswerable questions. These are designed to challenge models to not only extract answers from the provided context but also identify instances where no answer exists. This makes SQuADv2 a robust benchmark for assessing generalization, reasoning capabilities, and contextual understanding of large language models (LLMs). 

### MedMCQA Dataset
The MedMCQA (Medical Multiple-Choice Question Answering) dataset is a large-scale benchmark designed to evaluate models on medical question-answering tasks. It comprises over 300,000 multiple-choice questions spanning diverse medical disciplines such as pharmacology, physiology, anatomy, and pathology. Each question is accompanied by four answer options, with one correct answer, replicating the format of real-world medical examinations. The dataset poses domain-specific challenges by capturing the complexity of medical knowledge, requiring models to comprehend clinical context and apply medical reasoning. It includes a wide variety of question types, ranging from fact-based queries to scenario-driven reasoning, making it ideal for assessing a model's ability to generalize and solve problems effectively. Curated from publicly available medical resources, the dataset features a realistic distribution of topics and difficulty levels. 

Download all the .csv files in the website below
Website: https://www.kaggle.com/datasets/gvaldenebro/cancer-q-and-a-dataset

### Medical Corpus
General Medical articles. 79 articles were used.

## Conversion of Dataset
Run the "save_general_qa_as_json.py" to save the training and validation json file for General QA LoRA Model

Run the "save_medical_qa_as_json.py" to save the training and validation json file for Medical QA LoRA Model

## Training the LoRA models
Run the "PEFT_train_general_qa_lora.py" to train the medical General QA model

Run the "PEFT_train_medical_qa_lora.py" to train the medical Medical QA model

Run the "PEFT_train_autoregressive.py" to train the medical autoregressive model

Running any of the training scipts will save the base model (Gemma-2B), the LoRA model and the PEFT merged model in pytorch. A validation check also shows if the saved model has LoRA parameters. For the PEFT merged model, it is normal if the LoRA parameters do not exist as the LoRA parameters have been merged directly into the base model’s weights during the merging process.

## Combining the LoRA module into the base model
Run the "combine_general_qa_lora_and_base.py" to combine the medical General QA module with the base model

Run the "combine_medical_qa_lora_and_base.py" to combine the medical Medical QA model with the base model

Run the "combine_autoregressive_qa_lora_and_base.py" to combine the medical autoregressive model with the base model

Run the "combine_general_qa_lora_and_medical_autoregressive.py" to combine the General QA LoRA module and Medical Autoregressive LoRA Module with the base model
## Evaluating
