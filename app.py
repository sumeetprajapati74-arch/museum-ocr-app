import os
import re
import time
import streamlit as st
from PIL import Image
import google.generativeai as genai
from supabase import create_client

# Page Configuration
st.set_page_config(page_title="Museum Specimen Extractor", layout="wide")

# Secrets & Environment Variables Retrieval
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Please add it to Streamlit Secrets.")
    st.stop()

# Configure Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Supabase Client
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning(f"Could not connect to Supabase: {e}")

# Supported models to try sequentially
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

def gemini_call(prompt, image=None, retries=1):
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(retries + 1):
            try:
                model = genai.GenerativeModel(model_name)
                contents = [image, prompt] if image else [prompt]
                response = model.generate_content(contents)
                
                if response and response.text:
                    return {
                        "text": response.text.strip(),
                        "model": model_name,
                        "status": "SUCCESS",
                        "error": None
                    }
                last_error = "Empty response returned."
            except Exception as e:
                last_error = str(e)
                # If model is not found, jump directly to the next model in GEMINI_MODELS
                if "404" in last_error or "not found" in last_error.lower():
                    break
                if "429" in last_error or "quota" in last_error.lower():
                    break
                if attempt < retries:
                    time.sleep(2)
                    continue
                break

    return {
        "text": "",
        "model": None,
        "status": "FAILED",
        "error": last_error or "Unknown error"
    }

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
    return gemini_call(prompt, image=image)

def correct_ocr(ocr_text):
    prompt = f"""
Correct only obvious OCR mistakes in this museum specimen label text.
Do not invent text, translate, or add extra notes.
Preserve exact names, dates, place names, and numbers.

OCR TEXT:
{ocr_text}
"""
    return gemini_call(prompt)

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
    return gemini_call(prompt)

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

def save_to_supabase(image_name, ocr_text, corrected_text, date, locality, ver_status, ocr_m, corr_m, ext_m):
    if not supabase:
        return False, "Supabase client not initialized."
    try:
        data = {
            "image_name": image_name,
            "ocr_text": ocr_text,
            "corrected_text": corrected_text,
            "specimen_date": date,
            "locality": locality,
            "verification_status": ver_status,
            "ocr_model": ocr_m,
            "correction_model": corr_m,
            "extraction_model": ext_m
        }
        supabase.table("museum_specimens").insert(data).execute()
        return True, "Successfully saved record to Supabase database!"
    except Exception as e:
        return False, f"Failed to save to Supabase: {str(e)}"

def process_specimen(image, image_name="specimen_image.png"):
    ocr_result = run_ocr(image)
    if not ocr_result["text"]:
        return "", "", "REVIEW REQUIRED", f"Gemini OCR failed: {ocr_result['error']}", False, "OCR step failed."

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

    # Database Auto-Save Trigger
    save_status, save_msg = save_to_supabase(
        image_name=image_name,
        ocr_text=ocr_text,
        corrected_text=corrected_text,
        date=extracted_date,
        locality=extracted_locality,
        ver_status=verification_status,
        ocr_m=ocr_model,
        corr_m=correction_model,
        ext_m=extraction_model
    )

    return corrected_text, information, verification_status, "SUCCESS", save_status, save_msg

# Streamlit Interface
st.title("Museum Specimens Detection & Information Extraction System")
st.write("AI-powered museum specimen label processing using Gemini OCR.")

uploaded_file = st.file_uploader("Upload Museum Specimen Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Specimen Label", use_container_width=True)

    with col2:
        with st.spinner("Processing image via Gemini API..."):
            corrected_text, info, ver_status, proc_status, db_success, db_msg = process_specimen(image, uploaded_file.name)

        st.subheader("Results")
        st.text_input("Processing Status", value=proc_status, disabled=True)
        st.text_input("Verification Status", value=ver_status, disabled=True)
        st.text_area("Structured Information", value=info, height=150)
        st.text_area("Corrected OCR Text", value=corrected_text, height=200)

        if db_success:
            st.success(db_msg)
        else:
            st.warning(db_msg)
