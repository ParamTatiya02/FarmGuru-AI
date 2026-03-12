from pdf2image import convert_from_path
import pytesseract
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pdfplumber
import os

pdf_folder = "data"
DetectorFactory.seed = 0

# -------- SCRIPT DETECTION -------- #
def get_script_type(text: str, sample_size: int = 200) -> str:
    """
    Sample a small chunk + exit as soon as we're confident.
    No need to scan 3000 chars when 200 are enough.
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
def is_valid_text(text: str) -> bool:
    """
    Two checks:
    1. Character validity ratio (existing)
    2. NEW: Word sanity check — real text has common short words & vowels
    """
    if not text or len(text.strip()) < 50:
        return False

    # ---- Check 1: Character ratio (existing) ----
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

    ratio = readable / total

    # If Devanagari chars found → definitely valid, no need for word check
    if devanagari_count > 10:
        print(f"✅ Devanagari script detected directly")
        return True

    if ratio < 0.6:
        print(f"🔍 Text validity ratio: {ratio:.2f} → rejected")
        return False

    # ---- Check 2: Word sanity (catches garbled latin font text) ----
    words = text.lower().split()
    if not words:
        return False

    # Real English text always has vowels in most words
    vowels = set("aeiou")
    words_with_vowels = sum(1 for w in words if any(c in vowels for c in w))
    vowel_ratio = words_with_vowels / len(words)

    # Real English has common short connector words
    common_words = {"the", "is", "in", "of", "and", "to", "a", "for", "are", "with"}
    common_found = sum(1 for w in words if w in common_words)
    common_ratio = common_found / len(words)

    print(f"🔍 Validity ratio: {ratio:.2f} | Vowel ratio: {vowel_ratio:.2f} | Common words: {common_ratio:.2f}")

    # Garbled text has low vowel usage and almost no common English words
    return vowel_ratio > 0.6 and common_ratio > 0.02


# -------- LANGUAGE DETECTION -------- #
def detect_language(text: str) -> str:
    """
    Two-stage detection:
    1. Unicode range → identify script family
    2. langdetect → differentiate within same-script languages
    """
    if not text or not text.strip():
        return "unknown"

    sample = text[:3000]
    script = get_script_type(sample)

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
def extract_text_by_language(img, language: str) -> str:
    lang_map = {"mr": "mar", "hi": "hin", "en": "eng"}
    ocr_lang = lang_map.get(language, "eng")
    return pytesseract.image_to_string(img, lang=ocr_lang)


# -------- MAIN FUNCTION -------- #
def read_pdf(path: str) -> str:
    # Step 1: Try pdfplumber on first 3 pages only
    plumber_text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:3]:
            page_text = page.extract_text()
            if page_text:
                plumber_text += page_text

    # Step 2: Detect language — use only PAGE 1 for OCR if needed
    if is_valid_text(plumber_text):
        print("✅ pdfplumber text looks clean")
        language = detect_language(plumber_text[:3000])
    else:
        print("⚠️ Garbage text → using OCR on page 1 only for detection")
        first_page = convert_from_path(path, first_page=1, last_page=1)  # ✅ only 1 page
        sample = pytesseract.image_to_string(first_page[0], lang="eng+hin+mar")
        language = detect_language(sample)

    print(f"📄 Detected language: {language}")

    # Step 3: NOW convert and extract all pages
    all_images = convert_from_path(path)   # ✅ only called after language is confirmed
    print(f"📄 Total pages: {len(all_images)}")

    full_text = ""
    for page_num, img in enumerate(all_images):
        page_text = extract_text_by_language(img, language)
        full_text += page_text + "\n"
        print(f"✅ Page {page_num + 1}/{len(all_images)} done")

    return full_text


# -------- RUN -------- #
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"\n📂 Processing: {filename}")
        text = read_pdf(pdf_path)
