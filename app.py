# ============================================================
# 🐄 Cattle Ear-Tag Detection System
# Streamlit Cloud | EasyOCR + Tesseract | Multi-YOLO Support
# ============================================================

import os
import re
import cv2
import json
import zipfile
import tempfile
import numpy as np
import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
from ultralytics import YOLO
import easyocr
import io

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🐄 Cattle Ear-Tag AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0f1117;
        color: #e8e8e8;
    }
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #1b4332 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #40916c;
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        color: #d8f3dc;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #95d5b2;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        font-weight: 300;
    }
    .tag-card {
        background: #1e1e2e;
        border: 1px solid #2d6a4f;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .ocr-result {
        background: #0d1f1a;
        border-left: 3px solid #52b788;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        color: #d8f3dc;
        margin: 0.4rem 0;
    }
    .ocr-fail {
        border-left-color: #e63946;
        color: #ff8fa3;
    }
    .confidence-badge {
        display: inline-block;
        background: #1b4332;
        color: #95d5b2;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        border: 1px solid #40916c;
    }
    .metric-box {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #52b788;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-box .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stExpander"] {
        background: #16213e;
        border: 1px solid #2d6a4f;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background: #0d1f1a;
        border: 1px solid #40916c;
        color: #d8f3dc;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stSlider > div > div > div {
        color: #52b788;
    }
    .stDownloadButton > button {
        background: #2d6a4f;
        color: #d8f3dc;
        border: 1px solid #52b788;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background: #40916c;
        border-color: #74c69d;
    }
    .stButton > button {
        background: #1b4332;
        color: #d8f3dc;
        border: 1px solid #40916c;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] {
        background: #0d1f1a;
        border-right: 1px solid #1b4332;
    }
    .ocr-engine-tag {
        font-size: 0.65rem;
        background: #2d3561;
        color: #a8b4ff;
        padding: 1px 5px;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        margin-left: 5px;
    }
    .stSelectbox > div > div {
        background: #0d1f1a;
        border-color: #2d6a4f;
        color: #d8f3dc;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐄 Cattle Ear-Tag Detection System</h1>
    <p>YOLO Detection · EasyOCR + Tesseract · Multi-Model Support · Livestock ID Automation</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: Find all .pt models in repo root
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_yolo_models():
    """Scan directory for .pt model files."""
    models = []
    for f in sorted(os.listdir(BASE_DIR)):
        if f.endswith(".pt"):
            models.append(f)
    return models


# ─────────────────────────────────────────────
# Cached Loaders
# ─────────────────────────────────────────────
@st.cache_resource
def load_yolo(model_name: str):
    path = os.path.join(BASE_DIR, model_name)
    return YOLO(path)

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(
        ['en'],
        gpu=False,
        model_storage_directory="/tmp/easyocr",
        download_enabled=True
    )


# ─────────────────────────────────────────────
# OCR Preprocessing: Optimised for Numbers
# ─────────────────────────────────────────────
def preprocess_for_ocr(crop_bgr: np.ndarray) -> list:
    """
    Returns multiple preprocessed versions of the crop for best OCR coverage.
    Tuned to maximise numeric digit accuracy.
    """
    variants = []

    # 1. Upscale aggressively – OCR engines love larger text
    h, w = crop_bgr.shape[:2]
    scale = max(1, int(300 / min(h, w)))
    big = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # 3. Adaptive threshold (good for tags with varied lighting)
    adaptive = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 4. Otsu threshold (clean tags)
    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Inverted otsu (white digits on dark tags)
    otsu_inv = cv2.bitwise_not(otsu)

    # 6. Sharpened original
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel)

    variants.extend([clahe_img, adaptive, otsu, otsu_inv, sharpened])
    return variants


def clean_number_string(text: str) -> str:
    """
    Post-process OCR text to extract clean alphanumeric ear-tag IDs.
    Fixes common OCR confusions: O→0, I→1, l→1, S→5, B→8, Z→2.
    """
    if not text:
        return ""
    
    # Common OCR letter→digit confusions in tag context
    confusion_map = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2',
        'G': '6',
        'q': '9', 'g': '9',
        'D': '0',
    }
    
    # Keep letters, digits, hyphens (tag IDs like UK123456 or 4E-1234)
    cleaned = re.sub(r'[^A-Za-z0-9\-]', '', text)
    
    # Apply confusion correction only if string is mostly digits
    digits = sum(c.isdigit() for c in cleaned)
    letters = sum(c.isalpha() for c in cleaned)
    
    if digits > letters and letters > 0:
        result = ""
        for ch in cleaned:
            result += confusion_map.get(ch, ch)
        return result.upper()
    
    return cleaned.upper()


# ─────────────────────────────────────────────
# Dual OCR Engine
# ─────────────────────────────────────────────
def run_easyocr(reader, crop_bgr: np.ndarray) -> str:
    """Run EasyOCR across multiple preprocessed variants, pick best result."""
    variants = preprocess_for_ocr(crop_bgr)
    candidates = []

    for variant in variants:
        try:
            results = reader.readtext(variant, detail=1, paragraph=False,
                                      allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-')
            for (_, text, conf) in results:
                cleaned = clean_number_string(text)
                if cleaned and conf > 0.3:
                    candidates.append((cleaned, conf))
        except Exception:
            pass

    if not candidates:
        # Try original without allowlist
        try:
            raw = reader.readtext(crop_bgr, detail=0)
            text = " ".join(raw).strip()
            cleaned = clean_number_string(text)
            if cleaned:
                candidates.append((cleaned, 0.2))
        except Exception:
            pass

    if not candidates:
        return ""

    # Pick highest confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def run_tesseract(crop_bgr: np.ndarray) -> str:
    """Run Tesseract across multiple preprocessed variants."""
    variants = preprocess_for_ocr(crop_bgr)
    candidates = []

    # Tesseract configs tuned for tags:
    configs = [
        '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',  # single line
        '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',  # single word
        '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',  # uniform block
        '--oem 3 --psm 11',  # sparse text
    ]

    for variant in variants:
        pil_img = Image.fromarray(variant)
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(pil_img, config=cfg).strip()
                cleaned = clean_number_string(text)
                if cleaned and len(cleaned) >= 2:
                    # Get confidence data
                    data = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
                    confs = [int(c) for c in data['conf'] if str(c).isdigit() and int(c) >= 0]
                    avg_conf = sum(confs) / len(confs) if confs else 0
                    candidates.append((cleaned, avg_conf))
            except Exception:
                pass

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def dual_ocr(easyocr_reader, crop_bgr: np.ndarray, ocr_mode: str) -> tuple[str, str]:
    """
    Returns (final_text, engine_used).
    Runs both engines and picks the most credible result.
    """
    easy_result = ""
    tess_result = ""
    engine_used = ""

    if ocr_mode in ("EasyOCR Only", "Both (Best Result)"):
        easy_result = run_easyocr(easyocr_reader, crop_bgr)

    if ocr_mode in ("Tesseract Only", "Both (Best Result)"):
        tess_result = run_tesseract(crop_bgr)

    if ocr_mode == "EasyOCR Only":
        return easy_result, "EasyOCR"
    elif ocr_mode == "Tesseract Only":
        return tess_result, "Tesseract"
    else:
        # Both: prefer longer credible result, with tie-break logic
        def score(text):
            if not text:
                return -1
            digits = sum(c.isdigit() for c in text)
            return len(text) + digits  # reward more digit content

        if score(easy_result) >= score(tess_result) and easy_result:
            return easy_result, "EasyOCR"
        elif tess_result:
            return tess_result, "Tesseract"
        elif easy_result:
            return easy_result, "EasyOCR"
        return "", "—"


# ─────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # Model Selection
    st.markdown("### 🤖 YOLO Model")
    available_models = find_yolo_models()

    if not available_models:
        st.error("No `.pt` models found in repo root.\nAdd your YOLO `.pt` files and redeploy.")
        st.stop()

    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        help="All .pt files in the repo root appear here automatically."
    )

    st.divider()

    # Detection Settings
    st.markdown("### 🎯 Detection")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

    st.divider()

    # OCR Settings
    st.markdown("### 🔠 OCR Engine")
    ocr_mode = st.selectbox(
        "OCR Mode",
        ["Both (Best Result)", "EasyOCR Only", "Tesseract Only"],
        help=(
            "Both: runs EasyOCR + Tesseract and picks the best.\n"
            "Recommended for maximum accuracy on livestock tags."
        )
    )

    st.divider()

    # Info box
    st.markdown("""
    <div style='background:#0d1f1a; border:1px solid #2d6a4f; border-radius:8px; padding:1rem; font-size:0.78rem; color:#95d5b2;'>
    <b>💡 Tips for Best Results</b><br><br>
    • Use <b>Both</b> OCR mode for highest accuracy<br>
    • Lower confidence to catch faint tags<br>
    • EasyOCR is better for angled/blurry tags<br>
    • Tesseract excels on clean, upright text<br>
    • Correct any wrong IDs before downloading
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────
st.markdown("### 📂 Upload Images")
uploaded = st.file_uploader(
    "Drop a ZIP archive or individual image(s)",
    type=["zip", "jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="ZIP: multiple images in one archive. Single: one image at a time."
)

st.divider()


# ─────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────
if uploaded:

    # Load models
    with st.spinner(f"Loading YOLO model: `{selected_model}`…"):
        try:
            model = load_yolo(selected_model)
        except Exception as e:
            st.error(f"Failed to load `{selected_model}`: {e}")
            st.stop()

    with st.spinner("Initialising OCR engines…"):
        easyocr_reader = load_easyocr()

    # Collect images
    with tempfile.TemporaryDirectory() as tmp_dir:

        image_paths = []

        if uploaded.name.lower().endswith(".zip"):
            with zipfile.ZipFile(uploaded, "r") as z:
                z.extractall(tmp_dir)
            valid_exts = (".jpg", ".jpeg", ".png")
            for root, _, files in os.walk(tmp_dir):
                for f in sorted(files):
                    if f.lower().endswith(valid_exts) and not f.startswith("__"):
                        image_paths.append(os.path.join(root, f))
        else:
            save_path = os.path.join(tmp_dir, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            image_paths = [save_path]

        if not image_paths:
            st.error("No valid images found in the upload.")
            st.stop()

        # ── Metrics row ──────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value">{len(image_paths)}</div>
                <div class="metric-label">Images</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value">{selected_model.replace('.pt','')[:12]}</div>
                <div class="metric-label">Model</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value">{conf_threshold:.0%}</div>
                <div class="metric-label">Confidence</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value">{ocr_mode.split()[0]}</div>
                <div class="metric-label">OCR Mode</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        results_db = []
        total_tags = 0

        progress_bar = st.progress(0, text="Starting detection…")

        for img_idx, image_path in enumerate(image_paths, 1):

            img_name = os.path.basename(image_path)
            progress_bar.progress(
                img_idx / len(image_paths),
                text=f"Processing {img_name} ({img_idx}/{len(image_paths)})"
            )

            orig_img = cv2.imread(image_path)
            if orig_img is None:
                st.warning(f"⚠️ Could not read `{img_name}` — skipping.")
                continue

            # YOLO inference
            yolo_results = model(image_path, conf=conf_threshold, iou=iou_threshold)

            with st.expander(f"📷  {img_name}", expanded=True):

                left_col, right_col = st.columns([3, 2])

                with left_col:
                    annotated = yolo_results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="YOLO Detection", use_container_width=True)

                with right_col:
                    boxes = yolo_results[0].boxes

                    if boxes is None or len(boxes) == 0:
                        st.warning("No ear tags detected. Try lowering the confidence threshold.")
                        continue

                    st.markdown(f"**{len(boxes)} tag(s) detected**")

                    for tag_idx, box in enumerate(boxes, 1):
                        total_tags += 1
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Expand crop slightly for better OCR context
                        pad = 6
                        h_img, w_img = orig_img.shape[:2]
                        x1p = max(0, x1 - pad)
                        y1p = max(0, y1 - pad)
                        x2p = min(w_img, x2 + pad)
                        y2p = min(h_img, y2 + pad)

                        crop = orig_img[y1p:y2p, x1p:x2p]
                        if crop.size == 0:
                            continue

                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        # Show crop
                        st.image(crop_rgb, width=200, caption=f"Tag #{tag_idx} crop")

                        # Run OCR
                        with st.spinner(f"Reading tag #{tag_idx}…"):
                            ocr_text, engine = dual_ocr(easyocr_reader, crop, ocr_mode)

                        # Display OCR result
                        if ocr_text:
                            engine_badge = f'<span class="ocr-engine-tag">{engine}</span>'
                            st.markdown(
                                f'<div class="ocr-result">🔢 {ocr_text} {engine_badge}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="ocr-result ocr-fail">⚠️ OCR could not read tag</div>',
                                unsafe_allow_html=True
                            )

                        # Editable field
                        user_correction = st.text_input(
                            f"✏️ Correct Tag #{tag_idx}",
                            value=ocr_text,
                            key=f"edit_{img_idx}_{tag_idx}",
                            placeholder="Type correct ID if OCR is wrong"
                        )

                        final_id = user_correction.strip() if user_correction.strip() else ocr_text

                        # Confidence bar
                        st.progress(conf, text=f"Detection confidence: {conf:.1%}")
                        st.markdown("---")

                        results_db.append({
                            "image": img_name,
                            "tag_number": tag_idx,
                            "ocr_raw": ocr_text,
                            "ocr_engine": engine,
                            "user_corrected": final_id,
                            "detection_confidence": round(conf, 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "model_used": selected_model,
                            "timestamp": datetime.now().isoformat()
                        })

        progress_bar.empty()

        # ── Summary bar ──────────────────────────────
        if total_tags > 0:
            readable = sum(1 for r in results_db if r["ocr_raw"])
            st.success(f"✅ Scan complete — {total_tags} tag(s) found across {len(image_paths)} image(s). "
                       f"OCR read {readable}/{total_tags} successfully.")

        # ─────────────────────────────────────────────
        # Download Results
        # ─────────────────────────────────────────────
        if results_db:
            st.divider()
            st.markdown("### 💾 Download Results")

            dl1, dl2 = st.columns(2)

            with dl1:
                json_bytes = json.dumps(results_db, indent=2).encode("utf-8")
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_bytes,
                    file_name=f"eartag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            with dl2:
                csv_lines = ["Image,Tag#,OCR_Text,Engine,Final_ID,Confidence,Model,Timestamp"]
                for r in results_db:
                    csv_lines.append(
                        f'"{r["image"]}",'
                        f'{r["tag_number"]},'
                        f'"{r["ocr_raw"]}",'
                        f'{r["ocr_engine"]},'
                        f'"{r["user_corrected"]}",'
                        f'{r["detection_confidence"]:.4f},'
                        f'"{r["model_used"]}",'
                        f'{r["timestamp"]}'
                    )
                csv_bytes = "\n".join(csv_lines).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"eartag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            # Preview table
            with st.expander("📊 Results Preview"):
                preview = [
                    {
                        "Image": r["image"],
                        "Tag": r["tag_number"],
                        "Detected ID": r["ocr_raw"] or "—",
                        "Engine": r["ocr_engine"],
                        "Final ID": r["user_corrected"] or "—",
                        "Confidence": f"{r['detection_confidence']:.1%}",
                    }
                    for r in results_db
                ]
                st.table(preview)

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#555; border: 1px dashed #2d6a4f; border-radius:12px; background:#0d1a12;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>🐄</div>
        <div style='font-size:1.1rem; color:#95d5b2; font-family: IBM Plex Mono, monospace;'>
            Upload a ZIP archive or single image to begin
        </div>
        <div style='font-size:0.85rem; color:#555; margin-top:0.5rem;'>
            Supports JPG · JPEG · PNG · ZIP
        </div>
    </div>
    """, unsafe_allow_html=True)
