# Multi-LoRA-Merging
## About
This study explores whether combining two Low-Rank Adaptation (LoRA) models, trained on distinct tasks and domains, can achieve results comparable or superior to a single task- and domain-specific LoRA. Specifically, we examine the potential of merging a general question-answering LoRA model and a medical-trained autoregressive LoRA model to improve performance in a medical question-answering task. Additionally, we evaluate a third configuration where both domain and task align with a medical QA LoRA, demonstrating domain-specific adaptability. The primary objective is to determine if such a combination eliminates the need to train task-specific LoRAs for each domain, thereby increasing efficiency. We trained individual LoRA models on the SQuADv2 dataset, a medical QA dataset, and a medical autoregressive task using domain-specific documents. Subsequently, we manually merged the LoRA weights with the base model and compared the results with the PEFT-merged model to validate equivalence. Furthermore, we evaluated the combined general and domain-specific LoRA models to assess their performance against single task/domain-specific LoRAs

LoRA Paper: https://arxiv.org/abs/2106.09685

To evaluate the feasibility of integrating two LoRA modules, we extended the merging process to incorporate updates from both models sequentially: 

W' = W + (α1 / r1) ⋅ (B1 ⋅ A1) + (α2 / r2) ⋅ (B2 ⋅ A2)

where α1, r1 and α2, r2 are the scaling factors and ranks for the first and second LoRA models, respectively. The base model weights (W) are updated incrementally with the contributions of each LoRA module in a stepwise manner. The formula first applies the update from the first LoRA module (B1 ⋅ A1) scaled by (α1 / r1), modifying the base model weights to produce an intermediate state. Subsequently, the second module's update, (B2 ⋅ A2) scaled by (α2 / r2), is applied to this intermediate state, further refining the weights. Each step builds upon the results of the previous one.

## Environment
pip install -r requirements.txt

## Dataset
### SQuADv2 Dataset
The Stanford Question Answering Dataset (SQuADv2) is a widely used benchmark for evaluating question-answering systems. It comprises over 150,000 questions paired with context paragraphs extracted from Wikipedia, covering a broad range of topics. A key distinction from its predecessor, SQuADv1, is the inclusion of unanswerable questions. These are designed to challenge models to not only extract answers from the provided context but also identify instances where no answer exists. This makes SQuADv2 a robust benchmark for assessing generalization, reasoning capabilities, and contextual understanding of large language models (LLMs). 

### MedMCQA Dataset
The MedMCQA (Medical Multiple-Choice Question Answering) dataset is a large-scale benchmark designed to evaluate models on medical question-answering tasks. It comprises over 300,000 multiple-choice questions spanning diverse medical disciplines such as pharmacology, physiology, anatomy, and pathology. Each question is accompanied by four answer options, with one correct answer, replicating the format of real-world medical examinations. The dataset poses domain-specific challenges by capturing the complexity of medical knowledge, requiring models to comprehend clinical context and apply medical reasoning. It includes a wide variety of question types, ranging from fact-based queries to scenario-driven reasoning, making it ideal for assessing a model's ability to generalize and solve problems effectively. Curated from publicly available medical resources, the dataset features a realistic distribution of topics and difficulty levels. 

Download all the .csv files in the website below
Website: https://www.kaggle.com/datasets/gvaldenebro/cancer-q-and-a-dataset

### Medical Corpus
General Medical articles. 79 articles were used. Run the "extract_pdf_to_text.py" to extract text from PDF articles and save it into a text file for training and testing the Medical Autoregressive LoRA model.

## Conversion of Dataset
Run the "python3 save_general_qa_as_json.py" to save the training and validation json file for General QA LoRA Model

Run the "python3 save_medical_qa_as_json.py" to save the training and validation json file for Medical QA LoRA Model

## Training the LoRA models
Run the "python3 PEFT_train_general_qa_lora.py" to train the General QA LoRA module

Run the "python3 PEFT_train_medical_qa_lora.py" to train the Medical QA LoRA module

Run the "python3 PEFT_train_autoregressive.py" to train the autoregressive LoRA module

Running any of the training scipts will save the base model (Gemma-2B), the LoRA model and the PEFT merged model in pytorch. A validation check also shows if the saved model has LoRA parameters. For the PEFT merged model, it is normal if the LoRA parameters do not exist as the LoRA parameters have been merged directly into the base model’s weights during the merging process.

## Combining the LoRA module into the base model
Run the "python3 combine_general_qa_lora_and_base.py" to combine the General QA module with the base model

Run the "python3 combine_medical_qa_lora_and_base.py" to combine the Medical QA module with the base model

Run the "python3 combine_autoregressive_qa_lora_and_base.py" to combine the Medical Autoregressive module with the base model

Run the "python3 combine_general_qa_lora_and_medical_autoregressive.py" to combine the General QA LoRA module and Medical Autoregressive LoRA Module with the base model

## Evaluating
Run the "python3 evaluate_general_qa.py" to evaluate the General QA LoRA model

Run the "python3 evaluate_medical_qa.py" to evaluate the Medical QA LoRA model

Run the "python3 evaluate_autoregressive_model.py" to evaluate the Medical Autoregressive LoRA model

Run the "python3 evaluate_general_qa_medical_autoregressive.py" to combine the General QA LoRA module and Medical Autoregressive LoRA Module with the base model

## Discussion
While the Custom Combined General QA + Medical Autoregressive Model did not perform as well as the Medical QA Model, it did perform slightly better than the base model and demonstrated significant improvements over the individual General QA and Medical Autoregressive models on the Medical QA test dataset. Please refer to Appendix 2.
These results suggest that while the combined LoRA approach has potential, there are challenges in achieving substantial gains, likely due to issues inherent in the merging and fine-tuning processes.
Several potential causes could explain these results. One possible issue is catastrophic forgetting after merging, where previously learned knowledge is overwritten during the integration of multiple LoRA modules. Additionally, the merging weights for each LoRA model may not have been fully optimized, leading to suboptimal integration of task-specific knowledge. The General QA dataset itself could also contribute to the problem, as its answers may be too short to effectively train the model, while the medical documents may lack the depth or diversity needed to capture domain-specific nuances. Furthermore, conflicting knowledge from the models may arise when merging modules with different focuses, leading to interference between LoRA layers and diluting domain-specific knowledge.
Other factors include imbalanced training data, where one task might dominate the learning process, and the possibility of overfitting or underfitting, depending on the complexity of the tasks and the capacity of the merged model. Lastly, interference between LoRA layers could reduce the effectiveness of the domain-specific adaptations, further complicating the merging process. These potential issues highlight the need for more refined approaches to dataset preparation, LoRA merging strategies, and model optimization to fully realize the benefits of combining LoRA modules.

