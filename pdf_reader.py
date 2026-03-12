from pdf2image import convert_from_path
import pytesseract

images = convert_from_path(
    "data/आंबा फळपिक बुक.pdf"
)

for img in images:
    text = pytesseract.image_to_string(img, lang="mar")
    print(text)