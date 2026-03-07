```python
import streamlit as st
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import zipfile
import io
import re

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Cow Ear Tag Scanner",
    layout="wide"
)

# ==========================================================
# DARK MODE + RED SLIDER CSS
# ==========================================================

st.markdown("""
<style>

/* ---------------- DARK MODE ---------------- */

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #ffffff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161a23;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #161a23;
}

/* Divider */
hr {
    border-color: #333;
}

/* ---------------- SLIDER RED ---------------- */

/* Slider track */
div[data-baseweb="slider"] > div {
    background: #ff4b4b !important;
}

/* Filled portion */
div[data-baseweb="slider"] > div > div {
    background: #ff4b4b !important;
}

/* Slider handle */
div[role="slider"] {
    background: #ff4b4b !important;
    border: 3px solid #ff4b4b !important;
}

/* Slider label */
[data-testid="stSlider"] label {
    color: #ff4b4b !important;
    font-weight: bold;
}

/* ---------------- TAG BOX ---------------- */

.tag-id {
    font-size: 26px;
    font-weight: bold;
    color: #ff4b4b;
    background: #1e222c;
    padding: 12px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.title("🐄 Cow Ear Tag AI Scanner")
st.write("Upload a **single image** or a **ZIP file** containing cattle images.")

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():
    model = YOLO("cow_eartag_yolov8n_100ep_clean_best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, ocr_reader = load_models()

# ==========================================================
# UI CONTROLS
# ==========================================================

col1, col2 = st.columns([2,1])

with col1:
    uploaded_file = st.file_uploader(
        "📂 Upload Image or ZIP",
        type=["jpg","jpeg","png","zip"]
    )

with col2:
    conf_level = st.slider(
        "🎯 Detection Confidence",
        0.1, 1.0, 0.4
    )

# ==========================================================
# OCR MISHAP FIX MAP
# ==========================================================

MISHAP_MAP = {
    "|": "1",
    "I": "1",
    "l": "1",
    "[": "1",
    "]": "1",
    "(": "1",
    ")": "1",
    "O": "0",
    "o": "0",
    "S": "5",
    "s": "5",
    "B": "8",
    "G": "6"
}

# ==========================================================
# PROCESS FILES
# ==========================================================

if uploaded_file:

    images_to_process = []

    # ZIP SUPPORT
    if uploaded_file.name.endswith(".zip"):

        with zipfile.ZipFile(uploaded_file,"r") as z:
            for f in z.namelist():

                if f.lower().endswith(("jpg","jpeg","png")):

                    img_bytes = z.read(f)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images_to_process.append((f,img))

    else:

        img = Image.open(uploaded_file).convert("RGB")
        images_to_process.append((uploaded_file.name,img))

    # ======================================================
    # RUN DETECTION
    # ======================================================

    for img_name, img in images_to_process:

        st.markdown(f"### 📄 Processing `{img_name}`")

        results = detector(img, conf=conf_level)[0]

        annotated = results.plot()[:,:,::-1]

        st.image(annotated, use_container_width=True)

        boxes = results.boxes

        if boxes is None or len(boxes)==0:
            st.info("No tags detected in this image.")
            st.divider()
            continue

        # ==================================================
        # PROCESS EACH DETECTED TAG
        # ==================================================

        for i,box in enumerate(boxes):

            colA,colB = st.columns([1,2])

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            pad = 20

            tag_crop = img.crop((
                max(0,x1-pad),
                max(0,y1-pad),
                x2+pad,
                y2+pad
            ))

            # =========================
            # OCR PREPROCESS
            # =========================

            clean = ImageOps.grayscale(tag_crop)
            clean = ImageEnhance.Contrast(clean).enhance(2.0)
            clean = ImageOps.autocontrast(clean)

            # =========================
            # RUN OCR
            # =========================

            ocr_results = ocr_reader.readtext(np.array(clean))

            raw_text = "".join([r[1] for r in ocr_results]).strip()

            # =========================
            # FIX COMMON OCR ERRORS
            # =========================

            final_id = "".join(
                [MISHAP_MAP.get(c,c) for c in raw_text if c.isdigit() or c in MISHAP_MAP]
            )

            # =========================
            # DISPLAY
            # =========================

            with colA:
                st.image(tag_crop, caption=f"Tag #{i+1}")

            with colB:

                if final_id:
                    st.markdown(
                        f'<div class="tag-id">ID: {final_id}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Could not read ID")

                st.text(f"Raw detected: {raw_text}")

        st.divider()

else:

    st.info("📤 Upload an image or ZIP file to start scanning.")
```
