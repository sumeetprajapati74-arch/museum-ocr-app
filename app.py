```python
# ============================================================
# Museum Specimens Detection and Information Extraction System
# Deployment App
# ============================================================

import os
import io
import re
import time
import base64

import gradio as gr
from PIL import Image
from google import genai


# ============================================================
# 1. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Add it as a secret/environment variable in your deployment platform."
    )


# ============================================================
# 2. GEMINI FLASH MODELS
# ============================================================

GEMINI_FLASH_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


# ============================================================
# 3. GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# 4. READ GEMINI RESPONSE
# ============================================================

def get_response_text(response):

    try:
        if getattr(response, "output_text", None):
            return response.output_text.strip()
    except Exception:
        pass

    try:
        output = []

        for step in getattr(response, "steps", []) or []:

            if getattr(step, "type", None) == "model_output":

                for item in getattr(step, "content", []) or []:

                    text = getattr(item, "text", None)

                    if text:
                        output.append(text)

        return "\n".join(output).strip()

    except Exception:
        return ""


# ============================================================
# 5. UNIVERSAL GEMINI FLASH CALL
# ============================================================

def gemini_flash_call(
    prompt,
    image_b64=None,
    mime_type="image/jpeg",
    retries=1
):

    if image_b64 is None:

        model_input = prompt

    else:

        model_input = [
            {
                "type": "image",
                "data": image_b64,
                "mime_type": mime_type
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

    last_error = None

    for model in GEMINI_FLASH_MODELS:

        for attempt in range(retries + 1):

            try:

                print(
                    f"Trying Gemini model: {model} "
                    f"({attempt + 1}/{retries + 1})"
                )

                response = gemini_client.interactions.create(
                    model=model,
                    input=model_input
                )

                text = get_response_text(response)

                if text:

                    print(f"SUCCESS: {model}")

                    return {
                        "text": text,
                        "model": model,
                        "status": "SUCCESS",
                        "error": None
                    }

                last_error = "Gemini returned empty output."

            except Exception as e:

                last_error = e
                error_text = str(e)

                print(
                    f"{model} error: "
                    f"{error_text[:250]}"
                )

                # Project/API quota
                if (
                    "429" in error_text
                    or "quota" in error_text.lower()
                    or "resource exhausted" in error_text.lower()
                ):
                    break

                # Temporary server errors
                if any(
                    code in error_text
                    for code in ["500", "502", "503", "504"]
                ):

                    if attempt < retries:
                        time.sleep(3)
                        continue

                    break

                # Other errors
                break

    return {
        "text": "",
        "model": None,
        "status": "FAILED",
        "error": str(last_error) if last_error else "Unknown error"
    }


# ============================================================
# 6. PREPARE IMAGE
# ============================================================

def prepare_image(image, max_side=2500):

    image = image.convert("RGB")

    width, height = image.size

    if max(width, height) > max_side:

        scale = max_side / max(width, height)

        image = image.resize(
            (
                int(width * scale),
                int(height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# 7. OCR
# ============================================================

def run_ocr(image):

    image_b64 = prepare_image(image)

    prompt = """
Read ALL visible text on this museum specimen label.

Include:
- handwritten text
- printed text
- dates
- place names
- personal names
- numbers
- abbreviations

Preserve the spelling as closely as possible.

Do not guess missing text.

Return exactly in this format:

HANDWRITTEN:
<handwritten text>

PRINTED:
<printed text>
"""

    return gemini_flash_call(
        prompt,
        image_b64,
        mime_type="image/jpeg",
        retries=1
    )


# ============================================================
# 8. OCR CORRECTION
# ============================================================

def correct_ocr(ocr_text):

    prompt = f"""
Correct only obvious OCR mistakes in this museum specimen label.

Rules:
- Do not invent text.
- Preserve names.
- Preserve dates.
- Preserve place names.
- Preserve numbers.
- Preserve handwritten/printed information.
- Do not translate.
- Do not add explanations.

Return only the corrected text.

OCR TEXT:

{ocr_text}
"""

    return gemini_flash_call(
        prompt,
        retries=1
    )


# ============================================================
# 9. STRUCTURED DATE + LOCALITY EXTRACTION
# ============================================================

def extract_information(corrected_text):

    prompt = f"""
Extract DATE and LOCALITY from this museum specimen label.

Important rules:

1. Locality may appear anywhere in handwritten or printed text.
2. Do not require a word such as LOCALITY to be present.
3. A person's name is NOT automatically a locality.
4. Preserve exact spelling.
5. Preserve the date format.
6. Do not translate.
7. Do not guess.

Return EXACTLY:

DATE: <value or MISSING>
LOCALITY: <value or MISSING>

TEXT:

{corrected_text}
"""

    return gemini_flash_call(
        prompt,
        retries=1
    )


# ============================================================
# 10. LOCAL FALLBACK EXTRACTION
# ============================================================

KNOWN_LOCALITIES = [
    "Rösnæs",
    "Svinø strand",
    "Lodskovvad",
    "Dyrehaven",
    "Tisvilde",
    "Øbjerggård",
    "Bovbj.",
    "Turø Rev"
]


def fallback_extract(text):

    date = "MISSING"
    locality = "MISSING"

    # Date patterns
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

    # Known locality fallback
    for location in KNOWN_LOCALITIES:

        if location.lower() in text.lower():

            locality = location

            break

    return date, locality


# ============================================================
# 11. PARSE STRUCTURED RESPONSE
# ============================================================

def parse_information(structured_text):

    extracted_date = "MISSING"
    extracted_locality = "MISSING"

    if not structured_text:

        return extracted_date, extracted_locality

    for line in structured_text.splitlines():

        line = line.strip()

        if line.upper().startswith("DATE:"):

            extracted_date = line.split(
                ":",
                1
            )[1].strip()

        elif line.upper().startswith("LOCALITY:"):

            extracted_locality = line.split(
                ":",
                1
            )[1].strip()

    return extracted_date, extracted_locality


# ============================================================
# 12. MAIN PROCESSING FUNCTION
# ============================================================

def process_specimen(image):

    if image is None:

        return (
            "",
            "",
            "NO IMAGE",
            "Please upload a museum specimen image."
        )

    try:

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_result = run_ocr(image)

        if not ocr_result["text"]:

            return (
                "",
                "",
                "REVIEW REQUIRED",
                f"Gemini OCR failed.\n\n"
                f"Error: {ocr_result['error']}"
            )

        ocr_text = ocr_result["text"]

        ocr_model = ocr_result["model"]


        # ----------------------------------------------------
        # OCR CORRECTION
        # ----------------------------------------------------

        correction_result = correct_ocr(
            ocr_text
        )

        corrected_text = (
            correction_result["text"]
            or ocr_text
        )

        correction_model = (
            correction_result["model"]
        )


        # ----------------------------------------------------
        # DATE + LOCALITY EXTRACTION
        # ----------------------------------------------------

        extraction_result = extract_information(
            corrected_text
        )

        structured_text = (
            extraction_result["text"]
            or ""
        )

        extraction_model = (
            extraction_result["model"]
        )


        # ----------------------------------------------------
        # Parse Gemini extraction
        # ----------------------------------------------------

        extracted_date, extracted_locality = (
            parse_information(structured_text)
        )


        # ----------------------------------------------------
        # Local fallback
        # ----------------------------------------------------

        if (
            extracted_date == "MISSING"
            and extracted_locality == "MISSING"
        ):

            fallback_date, fallback_locality = (
                fallback_extract(corrected_text)
            )

            extracted_date = fallback_date
            extracted_locality = fallback_locality


        # ----------------------------------------------------
        # Verification status
        # ----------------------------------------------------

        if (
            extracted_date != "MISSING"
            and extracted_locality != "MISSING"
        ):

            verification_status = "AUTO VERIFIED"

        elif (
            extracted_date != "MISSING"
            or extracted_locality != "MISSING"
        ):

            verification_status = "PARTIAL REVIEW"

        else:

            verification_status = "REVIEW REQUIRED"


        # ----------------------------------------------------
        # Structured information
        # ----------------------------------------------------

        information = (
            f"DATE: {extracted_date}\n"
            f"LOCALITY: {extracted_locality}\n\n"
            f"OCR MODEL: {ocr_model}\n"
            f"CORRECTION MODEL: {correction_model}\n"
            f"EXTRACTION MODEL: {extraction_model}"
        )


        processing_status = (
            "SUCCESS"
        )


        return (
            corrected_text,
            information,
            verification_status,
            processing_status
        )


    except Exception as e:

        return (
            "",
            "",
            "ERROR",
            str(e)
        )


# ============================================================
# 13. GRADIO INTERFACE
# ============================================================

demo = gr.Interface(

    fn=process_specimen,

    inputs=gr.Image(
        type="pil",
        label="Upload Museum Specimen Image"
    ),

    outputs=[

        gr.Textbox(
            label="Corrected OCR Text",
            lines=14
        ),

        gr.Textbox(
            label="Structured Information",
            lines=10
        ),

        gr.Textbox(
            label="Verification Status"
        ),

        gr.Textbox(
            label="Processing Status",
            lines=4
        )
    ],

    title=(
        "Museum Specimens Detection "
        "and Information Extraction System"
    ),

    description=(
        "AI-powered museum specimen label processing "
        "using Gemini Flash OCR, OCR correction, "
        "and DATE/LOCALITY extraction."
    ),

    examples=None
)


# ============================================================
# 14. START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch()
```
