from pdf2image import convert_from_path
import pytesseract
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pdfplumber
import os

DetectorFactory.seed = 0


class PDFReader:
    """
    A multilingual PDF reader that supports English, Hindi, and Marathi.
    Automatically detects language and falls back to OCR for scanned/custom-font PDFs.
    """

    # Maps langdetect language codes to tesseract OCR language codes
    LANG_MAP = {"mr": "mar", "hi": "hin", "en": "eng"}

    def __init__(self, pdf_folder: str = "data"):
        self.pdf_folder = pdf_folder

    # -------- SCRIPT DETECTION -------- #
    def _get_script_type(self, text: str, sample_size: int = 200) -> str:
        """
        Identify script family using Unicode ranges.
        Samples a small chunk and exits early once confident.
        """
        devanagari_count = 0
        latin_count = 0
        threshold = 20

        step = max(1, len(text) // sample_size)
        sampled = text[::step][:sample_size]

        for char in sampled:
            cp = ord(char)
            if 0x0900 <= cp <= 0x097F:
                devanagari_count += 1
                if devanagari_count >= threshold:
                    return "devanagari"
            elif 0x0041 <= cp <= 0x007A:
                latin_count += 1
                if latin_count >= threshold:
                    return "latin"

        if devanagari_count > latin_count:
            return "devanagari"
        elif latin_count > 0:
            return "latin"
        return "unknown"

    # -------- TEXT VALIDATION -------- #
    def _is_valid_text(self, text: str) -> bool:
        """
        Two checks:
        1. Character validity ratio
        2. Word sanity check — real text has common short words & vowels
        Catches garbled text from custom Marathi/Hindi fonts.
        """
        if not text or len(text.strip()) < 50:
            return False

        readable = 0
        total = 0
        devanagari_count = 0

        for char in text:
            if char in (" ", "\n", "\t"):
                continue
            total += 1
            cp = ord(char)
            if 0x0900 <= cp <= 0x097F:
                readable += 1
                devanagari_count += 1
            elif 0x0020 <= cp <= 0x007E:
                readable += 1

        if total == 0:
            return False

        # Devanagari chars found → definitely valid
        if devanagari_count > 10:
            print("✅ Devanagari script detected directly")
            return True

        ratio = readable / total
        if ratio < 0.6:
            print(f"🔍 Text validity ratio: {ratio:.2f} → rejected")
            return False

        # Word sanity check for latin text
        words = text.lower().split()
        if not words:
            return False

        vowels = set("aeiou")
        vowel_ratio = sum(1 for w in words if any(c in vowels for c in w)) / len(words)

        common_words = {"the", "is", "in", "of", "and", "to", "a", "for", "are", "with"}
        common_ratio = sum(1 for w in words if w in common_words) / len(words)

        print(f"🔍 Validity ratio: {ratio:.2f} | Vowel ratio: {vowel_ratio:.2f} | Common words: {common_ratio:.2f}")
        return vowel_ratio > 0.6 and common_ratio > 0.02

    # -------- LANGUAGE DETECTION -------- #
    def _detect_language(self, text: str) -> str:
        """
        Two-stage detection:
        1. Unicode range → identify script family
        2. langdetect → differentiate within same-script languages
        """
        if not text or not text.strip():
            return "unknown"

        sample = text[:3000]
        script = self._get_script_type(sample)

        if script == "devanagari":
            try:
                lang = detect(sample)
                return lang if lang in ["hi", "mr"] else "hi"
            except LangDetectException:
                return "hi"

        elif script == "latin":
            try:
                lang = detect(sample)
                return lang if lang == "en" else "en"
            except LangDetectException:
                return "en"

        else:
            try:
                return detect(sample)
            except LangDetectException:
                return "unknown"

    # -------- OCR EXTRACTION -------- #
    def _extract_text_by_language(self, img, language: str) -> str:
        """Extract text from an image using the appropriate OCR language."""
        ocr_lang = self.LANG_MAP.get(language, "eng")
        return pytesseract.image_to_string(img, lang=ocr_lang)

    # -------- READ SINGLE PDF -------- #
    def read_pdf(self, path: str) -> str:
        """
        Read a single PDF file and return extracted text.
        Automatically detects language and uses OCR if needed.
        """
        # Step 1: Try pdfplumber on first 3 pages only (fast)
        plumber_text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    plumber_text += page_text

        # Step 2: Validate pdfplumber output
        if self._is_valid_text(plumber_text):
            print("✅ pdfplumber text looks clean")
            language = self._detect_language(plumber_text[:3000])
        else:
            print("⚠️  Garbage text → using OCR on page 1 only for detection")
            first_page = convert_from_path(path, first_page=1, last_page=1)
            sample = pytesseract.image_to_string(first_page[0], lang="eng+hin+mar")
            language = self._detect_language(sample)

        print(f"📄 Detected language: {language}")

        # Step 3: Convert and extract all pages
        all_images = convert_from_path(path)
        print(f"📄 Total pages: {len(all_images)}")

        full_text = ""
        for page_num, img in enumerate(all_images):
            full_text += self._extract_text_by_language(img, language) + "\n"

        print(f"✅ All {len(all_images)} pages extraction complete!")
        return full_text

    # -------- READ ALL PDFs IN FOLDER -------- #
    def read_all_pdfs(self) -> dict:
        """
        Read all PDFs in the folder.
        Returns a dict of { filename: extracted_text }
        """
        results = {}

        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            print(f"⚠️  No PDF files found in '{self.pdf_folder}'")
            return results

        print(f"📁 Found {len(pdf_files)} PDF(s) in '{self.pdf_folder}'\n")

        for filename in pdf_files:
            pdf_path = os.path.join(self.pdf_folder, filename)
            print(f"\n📂 Processing: {filename}")
            results[filename] = self.read_pdf(pdf_path)

        print(f"\n🎉 Done! Processed {len(results)} PDF(s)")
        return results


# -------- RUN -------- #
if __name__ == "__main__":
    reader = PDFReader(pdf_folder="data")
    all_texts = reader.read_all_pdfs()