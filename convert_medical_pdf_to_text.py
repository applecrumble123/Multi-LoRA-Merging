import os
from PyPDF2 import PdfReader

# Define paths
training_pdf_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/Medical Documents/train"
validation_pdf_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/Medical Documents/val"
training_text_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/Medical Documents/train_text"
validation_text_folder = "/mnt/c/Users/tohji/OneDrive/Desktop/Medical Documents/val_text"

# Ensure output folders exist
os.makedirs(training_text_folder, exist_ok=True)
os.makedirs(validation_text_folder, exist_ok=True)

# Function to convert PDF to text and save it
def convert_pdf_to_text(pdf_folder, text_folder):
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text_path = os.path.join(text_folder, f"{os.path.splitext(pdf_file)[0]}.txt")

            try:
                # Read the PDF
                reader = PdfReader(pdf_path)

                # Handle encrypted PDFs
                if reader.is_encrypted:
                    print(f"{pdf_file} is encrypted. Attempting decryption...")
                    # Replace 'your_password' with the actual password if known
                    if not reader.decrypt('your_password'):
                        print(f"Failed to decrypt {pdf_file}. Skipping...")
                        continue

                # Extract text
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text()

                # Save the text content to a .txt file
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text_content)

                print(f"Converted: {pdf_path} -> {text_path}")
            except Exception as e:
                print(f"Failed to convert {pdf_path}: {e}")

# Convert PDFs in both folders
convert_pdf_to_text(training_pdf_folder, training_text_folder)
convert_pdf_to_text(validation_pdf_folder, validation_text_folder)

print("Conversion completed!")
