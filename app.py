import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from ultralytics import YOLO
import zipfile, os, tempfile, re, io
import easyocr
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🐄 Cow Ear Tag AI", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .tag-box {
        background: #1e293b; color: #facc15;
        font-size: 2.8rem; font-weight: 900;
        letter-spacing: 6px; text-align: center;
        padding: 15px; border-radius: 12px;
        font-family: 'Courier New', monospace;
        border: 3px solid #facc15; margin: 10px 0;
    }
    .stMetric { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MISHAP MAPPING (Expanded for symbols and OCR errors)
# ─────────────────────────────────────────────────────────────
DIGIT_FIXES = {
    # The "1" family (Vertical marks)
    "|": "1", "l": "1", "I": "1", "i": "1", "!": "1", "[": "1", "]": "1", "/": "1", "\\": "1", "(": "1", ")": "1",
    # The "0" family
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0",
    # The "5" family
    "S": "5", "s": "5", "$": "5",
    # The "8" family
    "B": "8", "E": "8", "&": "8",
    # The "2" family
    "Z": "2", "z": "2",
    # The "6" family
    "G": "6", "b": "6",
    # The "9" family
    "g": "9", "q": "9",
    # The "7" family
    "T": "7", "t": "7", "f": "7",
    # The "4" family
    "A": "4", "H": "4", "Y": "4"
}

# ─────────────────────────────────────────────────────────────
# IMAGE HELPERS (PIL + NumPy only)
# ─────────────────────────────────────────────────────────────

def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def crop_box(img: Image.Image, x1, y1, x2, y2, pad=8) -> Image.Image:
    w, h = img.size
    return img.crop((
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ))

# ─────────────────────────────────────────────────────────────
# PREPROCESSING (Manual implementation of CLAHE & Otsu)
# ─────────────────────────────────────────────────────────────

def otsu_threshold(gray_arr: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray_arr.ravel(), bins=256, range=(0, 256))
    total = gray_arr.size
    sum_all = np.dot(np.arange(256), hist)
    best_thresh, best_var, w0, sum_bg = 0, 0.0, 0, 0
    for t in range(256):
        w0 += hist[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0: continue
        sum_bg += t * hist[t]
        mu0, mu1 = sum_bg / w0, (sum_all - sum_bg) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_thresh = var, t
    return np.where(gray_arr > best_thresh, 255, 0).astype(np.uint8)

def clahe_numpy(gray_arr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    h, w = gray_arr.shape
    th, tw = max(1, h // tile), max(1, w // tile)
    out = np.zeros_like(gray_arr, dtype=np.float32)
    for row in range(tile):
        for col in range(tile):
            r0, r1, c0, c1 = row*th, min((row+1)*th, h), col*tw, min((col+1)*tw, w)
            patch = gray_arr[r0:r1, c0:c1]
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            limit = int(clip_limit * patch.size / 256)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.clip(hist, 0, limit) + (excess // 256)
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min() if cdf[cdf > 0].size else 0
            lut = np.round((cdf - cdf_min) / max(patch.size - cdf_min, 1) * 255).astype(np.uint8)
            out[r0:r1, c0:c1] = lut[patch]
    return out.astype(np.uint8)

def preprocess_grayscale(crop_pil: Image.Image) -> Image.Image:
    gray = crop_pil.convert("L")
    w, h = gray.size
    # Fix scaling logic
    if min(w, h) < 60: scale = 5
    elif min(w, h) < 120: scale = 3
    else: scale = 1
    
    if scale > 1:
        gray = gray.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    
    arr = clahe_numpy(np.array(gray))
    pil = Image.fromarray(arr).filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    binary = otsu_threshold(np.array(pil))
    
    # Morphological closing via Scipy if available
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
        struct = np.ones((3,3))
        fg = (binary == 0)
        closed = binary_erosion(binary_dilation(fg, structure=struct), structure=struct)
        binary = np.where(closed, 0, 255).astype(np.uint8)
    except: pass
    
    return Image.fromarray(binary)

# ─────────────────────────────────────────────────────────────
# OCR & VALIDATION LOGIC
# ─────────────────────────────────────────────────────────────

def validate_number(text: str):
    if not text or text in ("UNREADABLE", "LOW CONFIDENCE"):
        return "???", "UNREADABLE", 0.0
    
    corrected, fixes = "", 0
    for ch in text.upper():
        if ch.isdigit():
            corrected += ch
        elif ch in DIGIT_FIXES:
            corrected += DIGIT_FIXES[ch]
            fixes += 1
            
    if not corrected: return "???", "UNREADABLE", 0.0
    
    status = "VERIFIED" if fixes == 0 else f"FIXED ({fixes})"
    conf = max(0.5, 0.98 - (fixes * 0.08))
    return corrected, status, round(conf, 2)

def _pick_dominant(results):
    if not results: return "", 0.0
    # Filter by detection height (Tallest = likely the printed ID)
    heights = [max(p[1] for p in b) - min(p[1] for p in b) for b, t, c in results]
    max_h = max(heights)
    dominant = [r for r, h in zip(results, heights) if h >= max_h * 0.6]
    dominant.sort(key=lambda x: np.mean([p[0] for p in x[0]]))
    text = "".join(re.sub(r"\D", "", r[1]) for r in dominant)
    conf = np.mean([r[2] for r in dominant]) if dominant else 0
    return text, conf

def ocr_crop(reader, crop_pil: Image.Image):
    # 1. Smart Trim: Remove 12% of borders to avoid tag edges being read as "|"
    w, h = crop_pil.size
    inner_crop = crop_pil.crop((int(w*0.12), int(h*0.12), int(w*0.88), int(h*0.88)))
    
    processed = preprocess_grayscale(inner_crop)
    inverted = ImageOps.invert(processed)
    
    best_text, best_conf = "", 0.0
    # Try 3 variants: Binary, Inverted, and Raw Inner Crop
    for img in [processed, inverted, inner_crop]:
        res = reader.readtext(np.array(img), allowlist="0123456789" + "".join(DIGIT_FIXES.keys()))
        t, c = _pick_dominant(res)
        if len(t) > len(best_text) or (len(t) == len(best_text) and c > best_conf):
            best_text, best_conf = t, c
            
    return best_text, best_conf

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    return YOLO("cow_eartag_yolov8n_100ep_clean_best.pt"), easyocr.Reader(['en'], gpu=False)

st.title("🐄 Livestock Ear Tag AI")
st.caption("Pure PIL/NumPy Pipeline — Optimized for Yellow Tags & Streamlit 3.14")

yolo_model, ocr_reader = load_models()

c1, c2, c3 = st.columns([2,1,1])
with c1:
    uploaded = st.file_uploader("Upload Image(s) or ZIP", type=["zip", "jpg", "png", "jpeg"])
with c2:
    conf_thresh = st.slider("Detection Confidence", 0.1, 1.0, 0.4)
with c3:
    padding = st.number_input("Tag Padding (px)", 0, 100, 30)

if uploaded:
    imgs_to_process = []
    if uploaded.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded, "r") as z:
            for f in z.namelist():
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    imgs_to_process.append(Image.open(io.BytesIO(z.read(f))).convert("RGB"))
    else:
        imgs_to_process.append(Image.open(uploaded).convert("RGB"))

    results_data = []

    for idx, img in enumerate(imgs_to_process):
        st.subheader(f"Image {idx+1}")
        # YOLO Detection
        yolo_res = yolo_model(img, conf=conf_thresh)[0]
        boxes = yolo_res.boxes
        
        # Manual plot (Convert YOLO BGR to RGB)
        plotted_np = yolo_res.plot()[:, :, ::-1]
        
        main_col, side_col = st.columns([2, 1])
        main_col.image(plotted_np, use_container_width=True, caption="YOLO Detections")
        
        with side_col:
            if len(boxes) == 0:
                st.info("No tags found.")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tag_crop = crop_box(img, x1, y1, x2, y2, pad=padding)
                
                raw_txt, ocr_conf = ocr_crop(ocr_reader, tag_crop)
                final_id, status, val_conf = validate_number(raw_txt)
                
                # Display individual tag UI
                st.markdown(f"**Tag #{i+1}**")
                st.image(tag_crop, width=180)
                st.markdown(f'<div class="tag-box">{final_id}</div>', unsafe_allow_html=True)
                st.caption(f"Status: {status} | Conf: {ocr_conf:.2%}")
                st.divider()
                
                results_data.append({
                    "Image": f"IMG_{idx+1}",
                    "Tag_Number": final_id,
                    "Raw_OCR": raw_txt,
                    "OCR_Confidence": f"{ocr_conf:.2f}",
                    "Status": status
                })

    if results_data:
        st.success("Processing Complete!")
        df = pd.DataFrame(results_data)
        st.subheader("📊 Session Results")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "tags.csv", "text/csv")

else:
    st.info("Please upload an image to begin.")
