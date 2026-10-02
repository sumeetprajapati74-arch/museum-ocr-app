# ============================================================
# Museum Specimens Detection and Information Extraction System
# Streamlit Deployment App
# ============================================================

import os
import re
import time
import streamlit as st
from PIL import Image
from google import genai

# ============================================================
# 1. PAGE SETUP & GEMINI API KEY
# ============================================================

st.set_page_config(
    page_title="Museum Specimen Extractor",
    layout="wide"
)

# Fetch API Key from Streamlit Secrets or Environment Variable
GEMINI_API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Please add it to Streamlit Secrets.")
    st.stop()

# ============================================================
# 2. GEMINI CLIENT & MODEL LIST
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_FLASH_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash"
]

# ============================================================
# 3. UNIVERSAL GEMINI CALL
# ============================================================

def gemini_flash_call(prompt, image=None, retries=1):
    contents = [image, prompt] if image else prompt
    last_error = None

    for model in GEMINI_FLASH_MODELS:
        for attempt in range(retries + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=contents
                )
                if response.text:
                    return {
                        "text": response.text.strip(),
                        "model": model,
                        "status": "SUCCESS",
                        "error": None
                    }
                last_error = "Gemini returned empty output."
            except Exception as e:
                last_error = e
                error_text = str(e)

                if "429" in error_text or "quota" in error_text.lower():
                    break

                if any(code in error_text for code in ["500", "502", "503", "504"]):
                    if attempt < retries:
                        time.sleep(2)
                        continue
                    break
                break

    return {
        "text": "",
        "model": None,
        "status": "FAILED",
        "error": str(last_error) if last_error else "Unknown error"
    }

# ============================================================
# 4. PROCESSING STAGES
# ============================================================

def run_ocr(image):
    prompt = """
Read ALL visible text on this museum specimen label.
Include handwritten text, printed text, dates, place names, personal names, numbers, and abbreviations.
Preserve spelling strictly.

Return format:
HANDWRITTEN:
<handwritten text>

PRINTED:
<printed text>
"""
    return gemini_flash_call(prompt, image=image)

def correct_ocr(ocr_text):
    prompt = f"""
Correct only obvious OCR mistakes in this museum specimen label text.
Do not invent text, translate, or add extra notes.
Preserve exact names, dates, place names, and numbers.

OCR TEXT:
{ocr_text}
"""
    return gemini_flash_call(prompt)

def extract_information(corrected_text):
    prompt = f"""
Extract DATE and LOCALITY from this museum specimen label.
Preserve exact spelling and format.

Return EXACTLY:
DATE: <value or MISSING>
LOCALITY: <value or MISSING>

TEXT:
{corrected_text}
"""
    return gemini_flash_call(prompt)

# ============================================================
# 5. LOCAL FALLBACK & PARSING
# ============================================================

KNOWN_LOCALITIES = [
    "Rösnæs", "Svinø strand", "Lodskovvad", "Dyrehaven",
    "Tisvilde", "Øbjerggård", "Bovbj.", "Turø Rev"
]

def fallback_extract(text):
    date = "MISSING"
    locality = "MISSING"

    date_patterns = [
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b"
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date = match.group(0)
            break

    for location in KNOWN_LOCALITIES:
        if location.lower() in text.lower():
            locality = location
            break

    return date, locality

def parse_information(structured_text):
    extracted_date = "MISSING"
    extracted_locality = "MISSING"

    if not structured_text:
        return extracted_date, extracted_locality

    for line in structured_text.splitlines():
        line = line.strip()
        if line.upper().startswith("DATE:"):
            extracted_date = line.split(":", 1)[1].strip()
        elif line.upper().startswith("LOCALITY:"):
            extracted_locality = line.split(":", 1)[1].strip()

    return extracted_date, extracted_locality

def process_specimen(image):
    ocr_result = run_ocr(image)
    if not ocr_result["text"]:
        return "", "", "REVIEW REQUIRED", f"Gemini OCR failed: {ocr_result['error']}"

    ocr_text = ocr_result["text"]
    ocr_model = ocr_result["model"]

    correction_result = correct_ocr(ocr_text)
    corrected_text = correction_result["text"] or ocr_text
    correction_model = correction_result["model"]

    extraction_result = extract_information(corrected_text)
    structured_text = extraction_result["text"] or ""
    extraction_model = extraction_result["model"]

    extracted_date, extracted_locality = parse_information(structured_text)

    if extracted_date == "MISSING" and extracted_locality == "MISSING":
        fallback_date, fallback_locality = fallback_extract(corrected_text)
        extracted_date = fallback_date
        extracted_locality = fallback_locality

    if extracted_date != "MISSING" and extracted_locality != "MISSING":
        verification_status = "AUTO VERIFIED"
    elif extracted_date != "MISSING" or extracted_locality != "MISSING":
        verification_status = "PARTIAL REVIEW"
    else:
        verification_status = "REVIEW REQUIRED"

    information = (
        f"DATE: {extracted_date}\n"
        f"LOCALITY: {extracted_locality}\n\n"
        f"OCR MODEL: {ocr_model}\n"
        f"CORRECTION MODEL: {correction_model}\n"
        f"EXTRACTION MODEL: {extraction_model}"
    )

    return corrected_text, information, verification_status, "SUCCESS"

# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.title("Museum Specimens Detection & Information Extraction System")
st.write("AI-powered museum specimen label processing using Gemini Flash OCR.")

uploaded_file = st.file_uploader("Upload Museum Specimen Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Specimen Label", use_container_width=True)

    with col2:
        with st.spinner("Processing image via Gemini API..."):
            corrected_text, info, ver_status, proc_status = process_specimen(image)

        st.subheader("Results")
        st.text_input("Processing Status", value=proc_status, disabled=True)
        st.text_input("Verification Status", value=ver_status, disabled=True)
        st.text_area("Structured Information", value=info, height=150)
        st.text_area("Corrected OCR Text", value=corrected_text, height=200)
