import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

# Example usage
pdf_path = "NGOTextFile.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
print(extracted_text)

# Translate 'billi se bhi baat karte' in English
translation = "cats also talk"
print(translation)
import nltk
from nltk.tokenize import word_tokenize

