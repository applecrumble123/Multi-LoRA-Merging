# Multi-LoRA-Merging
## About
This study explores whether combining two Low-Rank Adaptation (LoRA) models, trained on distinct tasks and domains, can achieve results comparable or superior to a single task- and domain-specific LoRA. Specifically, we examine the potential of merging a general question-answering LoRA model and a medical-trained autoregressive LoRA model to improve performance in a medical question-answering task. Additionally, we evaluate a third configuration where both domain and task align with a medical QA LoRA, demonstrating domain-specific adaptability. The primary objective is to determine if such a combination eliminates the need to train task-specific LoRAs for each domain, thereby increasing efficiency. We trained individual LoRA models on the SQuADv2 dataset, a medical QA dataset, and a medical autoregressive task using domain-specific documents. Subsequently, we manually merged the LoRA weights with the base model and compared the results with the PEFT-merged model to validate equivalence. Furthermore, we evaluated the combined general and domain-specific LoRA models to assess their performance against single task/domain-specific LoRAs

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
## Training

## Evaluating
