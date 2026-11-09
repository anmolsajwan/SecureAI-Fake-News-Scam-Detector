
from ocr_reader import extract_image_text

image_path = input("Enter image file path: ")

text = extract_image_text(image_path)

print("\nExtracted Image Text:\n")
print(text)

if text.strip():
    print("\nImage text extracted successfully!")
else:
    print("\nNo text found in image.")
