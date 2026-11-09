
from pdf_reader import extract_pdf_text

pdf_path = input("Enter PDF file path: ")

text = extract_pdf_text(pdf_path)

print("\nExtracted PDF Text:\n")
print(text[:2000])

if text.strip():
    print("\nPDF text extracted successfully!")
else:
    print("\nNo text found. This may be a scanned PDF.")
