import os
import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(page_title="Museum Specimen Detection & Extraction System", layout="centered")
st.title("Museum Specimen Detection & Information Extraction System")
st.write("Browser-based Gemini Flash OCR & Field Extraction Pipeline")

# Retrieve API key securely from Streamlit Secrets or Environment
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Supported Gemini models for google-genai SDK
GEMINI_FLASH_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

def gemini_flash_api(client, prompt, image=None):
    last_err = None
    for model in GEMINI_FLASH_MODELS:
        try:
            contents = [prompt]
            if image is not None:
                contents.insert(0, image)

            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            return {"text": response.text.strip(), "model": model, "status": "SUCCESS", "error": None}
        except Exception as e:
            last_err = str(e)
            continue

    return {"text": "", "model": None, "status": "FAILED", "error": last_err or "API Request Failed"}

uploaded_file = st.file_uploader("Upload Museum Specimen Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Specimen Image", use_container_width=True)

    if st.button("Process Specimen Label", type="primary"):
        if not api_key:
            st.error("API Key missing! Please configure GEMINI_API_KEY in Streamlit Secrets.")
        else:
            with st.spinner("Running Gemini Multi-Step OCR & Field Extraction Pipeline..."):
                try:
                    client = genai.Client(api_key=api_key)

                    # Step A: OCR
                    ocr_prompt = "Read all visible text from this museum specimen label. Include handwritten and printed text. Return HANDWRITTEN: and PRINTED: sections."
                    r = gemini_flash_api(client, ocr_prompt, image=image)

                    if not r["text"]:
                        st.error(f"OCR Error: {r['error']}")
                    else:
                        # Step B: Correction
                        corr_prompt = f"Correct obvious OCR mistakes. Preserve names, dates and localities. Return only corrected text.\n\n{r['text']}"
                        corrected = gemini_flash_api(client, corr_prompt)
                        txt = corrected["text"] or r["text"]

                        # Step C: Extraction
                        ext_prompt = f"Extract DATE and LOCALITY. Return exactly DATE: <value or MISSING> and LOCALITY: <value or MISSING>.\n\n{txt}"
                        ext = gemini_flash_api(client, ext_prompt)

                        d = "MISSING"
                        l = "MISSING"
                        for line in ext["text"].splitlines():
                            if line.upper().startswith("DATE:"):
                                d = line.split(":", 1)[1].strip()
                            elif line.upper().startswith("LOCALITY:"):
                                l = line.split(":", 1)[1].strip()

                        status = "AUTO VERIFIED" if (d != "MISSING" and l != "MISSING") else ("PARTIAL REVIEW" if (d != "MISSING" or l != "MISSING") else "REVIEW REQUIRED")

                        # Display Results
                        st.subheader("Results")
                        st.text_area("Corrected OCR Text", value=txt, height=150)
                        
                        st.subheader("Extracted Structured Information")
                        info_text = f"DATE: {d}\nLOCALITY: {l}\nOCR MODEL: {r['model']}\nCORRECTION MODEL: {corrected['model']}\nEXTRACTION MODEL: {ext['model']}"
                        st.text_area("Field Extraction", value=info_text, height=120)

                        st.info(f"Verification Status: {status}")
                        st.success("Processing Status: SUCCESS")

                except Exception as e:
                    st.error(f"Execution Error: {e}")
