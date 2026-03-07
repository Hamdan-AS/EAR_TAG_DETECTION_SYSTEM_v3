import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from ultralytics import YOLO
import zipfile, os, tempfile, re, io
import easyocr
import pandas as pd
from datetime import datetime
from scipy.ndimage import binary_dilation, binary_erosion

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🐄 Cow Ear Tag AI", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .tag-box {
        background: #0f172a; color: #fbbf24;
        font-size: 2.8rem; font-weight: 900;
        letter-spacing: 6px; text-align: center;
        padding: 15px; border-radius: 12px;
        font-family: 'Courier New', monospace;
        border: 3px solid #fbbf24; margin: 10px 0;
    }
    .stExpander { border: 1px solid #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MISHAP MAPPING (Correcting common OCR errors)
# ─────────────────────────────────────────────────────────────
DIGITS_ONLY = "0123456789"
DIGIT_FIXES = {
    # The "1" family
    "|": "1", "I": "1", "l": "1", "i": "1", "!": "1", "[": "1", "]": "1", "/": "1", "\\": "1",
    # The "0" family
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0", "C": "0",
    # The "5" family
    "S": "5", "s": "5", "$": "5",
    # The "8" family
    "B": "8", "&": "8",
    # The "2" family
    "Z": "2", "z": "2",
    # The "6" & "9" family
    "G": "6", "b": "6", "g": "9", "q": "9",
    # Others
    "T": "7", "t": "7", "A": "4", "H": "4"
}

# ─────────────────────────────────────────────────────────────
# IMAGE HELPERS (PIL + NumPy only)
# ─────────────────────────────────────────────────────────────

def pil_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def crop_box(img: Image.Image, x1, y1, x2, y2, pad=15) -> Image.Image:
    """Crop with padding, clamped to image boundaries."""
    w, h = img.size
    return img.crop((
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ))

def otsu_threshold(gray_arr: np.ndarray) -> np.ndarray:
    """Pure-numpy Otsu binarization for tag contrast."""
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
    """Contrast-Limited Adaptive Histogram Equalization (Numpy implementation)."""
    h, w = gray_arr.shape
    th, tw = max(1, h // tile), max(1, w // tile)
    out = np.zeros_like(gray_arr, dtype=np.float32)
    for row in range(tile):
        for col in range(tile):
            r0, r1 = row * th, min((row + 1) * th, h)
            c0, c1 = col * tw, min((col + 1) * tw, w)
            patch = gray_arr[r0:r1, c0:c1]
            hist, bins = np.histogram(patch, bins=256, range=(0, 256))
            limit = int(clip_limit * patch.size / 256)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + (excess // 256)
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
            out[r0:r1, c0:c1] = cdf[patch]
    return out.astype(np.uint8)

def preprocess_grayscale(crop_pil: Image.Image) -> Image.Image:
    """Enhanced pipeline for livestock ear-tag numerals."""
    gray = crop_pil.convert("L")
    w, h = gray.size
    
    # Auto-scaling
    scale = 1
    if min(w, h) < 60: scale = 4
    elif min(w, h) < 150: scale = 2
    if scale > 1:
        gray = gray.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    
    arr = clahe_numpy(np.array(gray))
    # Sharpening
    pil = Image.fromarray(arr).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    binary_arr = otsu_threshold(np.array(pil))
    
    # Morphological Close (Fills gaps in printed numbers)
    fg = binary_arr == 0
    closed = binary_erosion(binary_dilation(fg, iterations=1), iterations=1)
    return Image.fromarray(np.where(closed, 0, 255).astype(np.uint8))

# ─────────────────────────────────────────────────────────────
# LOGIC: DOMINANT NUMBER & VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_number(text: str):
    if not text or text in ("UNREADABLE", "LOW"):
        return text, "UNREADABLE", 0.0
    corrected, fixes = "", 0
    for ch in text.upper():
        if ch.isdigit(): corrected += ch
        elif ch in DIGIT_FIXES:
            corrected += DIGIT_FIXES[ch]
            fixes += 1
    if not corrected: return text, "UNREADABLE", 0.0
    status = "VERIFIED" if fixes == 0 else f"FIXED ({fixes})"
    return corrected, status, round(max(0.5, 0.98 - (fixes * 0.08)), 2)

def pick_dominant_number(results: list):
    """Filters out small handwritten text, keeps large printed IDs."""
    valid = []
    for (bbox, text, conf) in results:
        clean = re.sub(r"[^A-Z0-9|!\[\]/]", "", text.upper())
        if clean and float(conf) > 0.20:
            h = max([pt[1] for pt in bbox]) - min([pt[1] for pt in bbox])
            valid.append((bbox, text, float(conf), h))
    if not valid: return "", 0.0
    
    max_h = max(v[3] for v in valid)
    # Only keep text blocks that are at least 60% as tall as the largest block
    dominant = [v for v in valid if v[3] >= max_h * 0.6]
    dominant.sort(key=lambda x: np.mean([pt[0] for pt in x[0]])) # Sort Left-to-Right
    
    merged = "".join(v[1] for v in dominant)
    avg_conf = np.mean([v[2] for v in dominant])
    return merged, avg_conf

def ocr_crop(reader, crop_pil: Image.Image):
    processed = preprocess_grayscale(crop_pil)
    inverted = ImageOps.invert(processed)
    candidates = [np.array(processed), np.array(inverted), np.array(crop_pil)]
    
    best_t, best_c = "", 0.0
    for img_arr in candidates:
        try:
            res = reader.readtext(img_arr, decoder="beamsearch", beamWidth=10, mag_ratio=1.5)
            text, conf = pick_dominant_number(res)
            if len(text) > len(best_t) or (len(text) == len(best_t) and conf > best_c):
                best_t, best_c = text, conf
        except: continue
    return best_t or "UNREADABLE", round(best_c, 4)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    y = YOLO("cow_eartag_yolov8n_100ep_clean_best.pt") # Ensure this file exists
    r = easyocr.Reader(["en"], gpu=False)
    return y, r

st.title("🐄 Cattle Ear Tag AI")
st.caption("Detection + OCR | Powered by YOLOv8 & EasyOCR | OpenCV-Free")

yolo_model, ocr_reader = load_models()

uploaded = st.file_uploader("Upload Cattle Image or ZIP", type=["jpg","jpeg","png","zip"])

if uploaded:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        if uploaded.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded, "r") as z:
                z.extractall(tmp)
            paths = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.lower().endswith((".jpg",".png",".jpeg"))]
        else:
            p = os.path.join(tmp, uploaded.name)
            with open(p, "wb") as f: f.write(uploaded.getbuffer())
            paths = [p]

        final_data = []
        for p in sorted(paths):
            img_pil = pil_from_path(p)
            results = yolo_model(p, conf=0.4)
            boxes = results[0].boxes
            
            with st.expander(f"🖼️ {os.path.basename(p)} - {len(boxes)} tags found", expanded=True):
                c_main, c_tags = st.columns([2, 1])
                
                # Full Annotated Image (Convert YOLO BGR to RGB)
                annotated = results[0].plot()[:, :, ::-1]
                c_main.image(annotated, use_container_width=True)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    tag_crop = crop_box(img_pil, x1, y1, x2, y2)
                    
                    # Run OCR
                    raw_text, conf = ocr_crop(ocr_reader, tag_crop)
                    clean_id, status, val_conf = validate_number(raw_text)
                    
                    with c_tags:
                        st.image(tag_crop, caption=f"Tag #{i+1}")
                        st.markdown(f'<div class="tag-box">{clean_id if clean_id else "???"}</div>', unsafe_allow_html=True)
                        st.caption(f"Raw: {raw_text} | {status}")
                        st.divider()
                    
                    final_data.append({
                        "Image": os.path.basename(p),
                        "Tag_ID": clean_id,
                        "Status": status,
                        "OCR_Conf": conf,
                        "Timestamp": datetime.now().strftime("%H:%M:%S")
                    })

        if final_data:
            st.subheader("📊 Session Results")
            df = pd.DataFrame(final_data)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
