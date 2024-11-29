import pdfplumber
import os 
pdf_path = 'path/to/your/file.pdf'  # Replace with your PDF path


root_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/"
pdf_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/Davidson's_Principles_and_Practice_of_Medicine_24th_Ed_Full_version.pdf")
text_file_path = os.path.join(root_folder, "add_multiple_lora_to_base_model/medical_document_5.txt")

with pdfplumber.open(pdf_path) as pdf:
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_file.write(text)
                text_file.write('\n')  # Add a newline after each page
            
print(f'Text successfully saved to {text_file_path}')
