import streamlit as st
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import re

# 1. SETUP & MODELS
st.set_page_config(page_title="Cow Tag Scanner", layout="wide")
st.title("🐄 Cow Ear Tag AI")
st.write("A simple demo to detect tags and read ID numbers using YOLOv8 & EasyOCR.")

# Dictionary to fix common OCR mishaps (like | being read as 1)
MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

@st.cache_resource
def load_models():
    # Load YOLO (Detection) and EasyOCR (Reading)
    model = YOLO("cow_eartag_yolov8n_100ep_clean_best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, ocr_reader = load_models()

# 2. SIDEBAR SETTINGS
st.sidebar.header("Settings")
conf_level = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)
padding = st.sidebar.number_input("Crop Padding (px)", 0, 100, 20)

# 3. UPLOAD IMAGE
uploaded_file = st.file_uploader("Choose a cattle image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image with PIL
    img = Image.open(uploaded_file).convert("RGB")
    
    # RUN YOLO DETECTION
    results = detector(img, conf=conf_level)[0]
    
    # Show the full image with boxes
    # Note: .plot() returns a BGR numpy array, we flip it to RGB for Streamlit
    annotated_img = results.plot()[:, :, ::-1]
    st.image(annotated_img, caption="Detected Tags", use_container_width=True)

    # 4. PROCESS EACH DETECTION
    boxes = results.boxes
    if len(boxes) > 0:
        st.subheader(f"Found {len(boxes)} tag(s):")
        
        # Create a row for each detected tag
        for i, box in enumerate(boxes):
            cols = st.columns([1, 2])
            
            # Get coordinates and crop
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Add padding manually
            tag_crop = img.crop((x1-padding, y1-padding, x2+padding, y2+padding))
            
            # PRE-PROCESS FOR OCR (Make it easier for the AI)
            # 1. Grayscale 2. Boost Contrast 3. Auto-level
            clean_crop = ImageOps.grayscale(tag_crop)
            clean_crop = ImageEnhance.Contrast(clean_crop).enhance(2.0)
            clean_crop = ImageOps.autocontrast(clean_crop)

            # RUN OCR
            # Convert PIL to Numpy for EasyOCR
            crop_np = np.array(clean_crop)
            ocr_results = ocr_reader.readtext(crop_np)
            
            # Extract text and fix mishaps
            raw_text = "".join([res[1] for res in ocr_results]).strip()
            
            # Apply our MISHAP_MAP fixes
            final_id = ""
            for char in raw_text:
                if char.isdigit():
                    final_id += char
                elif char in MISHAP_MAP:
                    final_id += MISHAP_MAP[char]

            # 5. DISPLAY RESULTS
            with cols[0]:
                st.image(tag_crop, caption=f"Tag #{i+1}")
            
            with cols[1]:
                if final_id:
                    st.success(f"**Read ID: {final_id}**")
                else:
                    st.warning("Could not read numbers clearly.")
                
                # Show the raw OCR result for debugging
                st.text(f"Raw OCR output: {raw_text}")
            st.divider()
    else:
        st.info("No tags detected in this image. Try lowering the confidence slider.")

else:
    st.info("Please upload an image to start the detection demo.")

# 6. FOOTER
st.markdown("---")
st.caption("Built with Streamlit, YOLOv8, and EasyOCR. (No OpenCV version)")
