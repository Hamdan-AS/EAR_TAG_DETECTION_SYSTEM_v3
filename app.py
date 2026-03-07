import streamlit as st
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import zipfile, io, re

# 1. PAGE SETUP
st.set_page_config(page_title="Cow Tag Scanner", layout="wide")

# CSS to turn the slider RED and style the UI
st.markdown("""
    <style>
    /* Turn the slider track and handle RED */
    div[data-baseweb="slider"] > div > div { background-color: #ff4b4b !important; }
    div[role="slider"] { background-color: #ff4b4b !important; border-color: #ff4b4b !important; }
    .tag-id { font-size: 24px; font-weight: bold; color: #ff4b4b; background: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🐄 Cow Ear Tag AI")
st.write("Upload an Image or a **ZIP file** to detect tags and read IDs.")

# 2. LOAD MODELS
@st.cache_resource
def load_models():
    model = YOLO("cow_eartag_yolov8n_100ep_clean_best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, ocr_reader = load_models()

# 3. MAIN PAGE CONTROLS (DETECTION CONFIDENCE IN RED)
col_ui1, col_ui2 = st.columns([2, 1])
with col_ui1:
    uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg", "jpeg", "png", "zip"])
with col_ui2:
    conf_level = st.slider("🎯 Detection Confidence (Red)", 0.1, 1.0, 0.4)

# MISHAP MAPPING
MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

# 4. PROCESSING LOGIC
if uploaded_file:
    # Handle ZIP or Single Image
    images_to_process = []
    
    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file, "r") as z:
            for f in z.namelist():
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_data = Image.open(io.BytesIO(z.read(f))).convert("RGB")
                    images_to_process.append((f, img_data))
    else:
        images_to_process.append((uploaded_file.name, Image.open(uploaded_file).convert("RGB")))

    # Process all images found
    for img_name, img in images_to_process:
        st.markdown(f"### 📄 Processing: `{img_name}`")
        
        # YOLO DETECTION
        results = detector(img, conf=conf_level)[0]
        
        # Show annotated image
        annotated_img = results.plot()[:, :, ::-1]
        st.image(annotated_img, use_container_width=True)

        boxes = results.boxes
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                c1, c2 = st.columns([1, 2])
                
                # Crop and Pad
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad = 20
                tag_crop = img.crop((x1-pad, y1-pad, x2+pad, y2+pad))
                
                # PRE-PROCESS FOR OCR
                clean_crop = ImageOps.grayscale(tag_crop)
                clean_crop = ImageEnhance.Contrast(clean_crop).enhance(2.0)
                clean_crop = ImageOps.autocontrast(clean_crop)

                # RUN OCR
                ocr_results = ocr_reader.readtext(np.array(clean_crop))
                raw_text = "".join([res[1] for res in ocr_results]).strip()
                
                # Fix mishaps (| -> 1, etc.)
                final_id = "".join([MISHAP_MAP.get(c, c) for c in raw_text if c.isdigit() or c in MISHAP_MAP])

                with c1:
                    st.image(tag_crop, caption=f"Tag #{i+1}")
                with c2:
                    if final_id:
                        st.markdown(f'<div class="tag-id">ID: {final_id}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Could not read ID")
                    st.text(f"Raw detected: {raw_text}")
        else:
            st.info("No tags detected in this image.")
        st.divider()

else:
    st.info("Waiting for upload... You can drop a ZIP of cattle photos here.")
