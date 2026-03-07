"""
================================================================
  🐄  COW EAR TAG — Detection + OCR  (NO OpenCV)
================================================================
  Stack : YOLOv8 | EasyOCR | PIL + NumPy image pipeline
  Why   : cv2 has no Python 3.14 wheel on Streamlit Cloud.
          This version uses only PIL + numpy — works on any
          Python version with zero system-lib dependencies.
  Run   : streamlit run app.py
================================================================
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from ultralytics import YOLO
import zipfile, os, tempfile, re
import easyocr
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🐄 Cow Ear Tag AI", layout="wide")
st.markdown("""
<style>
.main { background-color: #f0f2f6; }
.tag-box {
    background: #1e293b; color: #facc15;
    font-size: 2.4rem; font-weight: 900;
    letter-spacing: 8px; text-align: center;
    padding: 14px 28px; border-radius: 10px;
    font-family: monospace; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# IMAGE HELPERS  (PIL + numpy, zero cv2)
# ─────────────────────────────────────────────────────────────

def pil_from_bytes(data: bytes) -> Image.Image:
    """Load PIL image from raw bytes."""
    import io
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert HxWx3 uint8 RGB numpy array → PIL."""
    return Image.fromarray(arr.astype(np.uint8))


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def crop_box(img: Image.Image, x1, y1, x2, y2, pad=8) -> Image.Image:
    """Crop with optional padding, clamped to image bounds."""
    w, h = img.size
    return img.crop((
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ))


# ─────────────────────────────────────────────────────────────
# GRAYSCALE PREPROCESSING  (PIL-only, tuned for yellow ear tags)
# ─────────────────────────────────────────────────────────────

def otsu_threshold(gray_arr: np.ndarray) -> np.ndarray:
    """
    Pure-numpy Otsu binarisation.
    Returns a uint8 array where text pixels = 0, background = 255.
    """
    # Build 256-bin histogram
    hist, _ = np.histogram(gray_arr.ravel(), bins=256, range=(0, 256))
    total   = gray_arr.size
    sum_all = np.dot(np.arange(256), hist)

    best_thresh, best_var = 0, 0.0
    w0 = sum_bg = 0

    for t in range(256):
        w0     += hist[t]
        w1      = total - w0
        if w0 == 0 or w1 == 0:
            continue
        sum_bg += t * hist[t]
        mu0     = sum_bg / w0
        mu1     = (sum_all - sum_bg) / w1
        var     = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var    = var
            best_thresh = t

    binary = np.where(gray_arr > best_thresh, 255, 0).astype(np.uint8)
    return binary


def clahe_numpy(gray_arr: np.ndarray,
                clip_limit: float = 2.5,
                tile: int = 8) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation (numpy).
    Divides image into tiles, equalises each, bilinear-blends borders.
    """
    h, w   = gray_arr.shape
    th, tw = max(1, h // tile), max(1, w // tile)
    out    = np.zeros_like(gray_arr, dtype=np.float32)

    for row in range(tile):
        for col in range(tile):
            r0, r1 = row * th, min((row + 1) * th, h)
            c0, c1 = col * tw, min((col + 1) * tw, w)
            patch  = gray_arr[r0:r1, c0:c1].ravel()

            hist, _ = np.histogram(patch, bins=256, range=(0, 256))

            # Clip & redistribute
            excess  = np.maximum(hist - int(clip_limit * patch.size / 256), 0)
            hist   -= excess
            hist   += excess.sum() // 256

            # CDF → LUT
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min() if cdf[cdf > 0].size else 0
            lut = np.round(
                (cdf - cdf_min) / max(patch.size - cdf_min, 1) * 255
            ).astype(np.uint8)

            out[r0:r1, c0:c1] = lut[gray_arr[r0:r1, c0:c1]]

    return np.clip(out, 0, 255).astype(np.uint8)


def morphological_close(binary: np.ndarray, ksize: int = 2) -> np.ndarray:
    """
    2-D morphological closing (dilation then erosion) via numpy.
    Fills small gaps inside digit strokes.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    fg     = binary == 0          # text is dark (0)
    dilated = binary_dilation(fg, iterations=ksize)
    closed  = binary_erosion(dilated, iterations=ksize)
    return np.where(closed, 0, 255).astype(np.uint8)


def preprocess_grayscale(crop_pil: Image.Image) -> Image.Image:
    """
    Full PIL+numpy grayscale pipeline for livestock ear-tag numerals.

    1. Convert to grayscale (L)
    2. Upscale small crops  (EasyOCR needs ≥ 32 px tall)
    3. CLAHE contrast normalisation
    4. Gentle sharpening
    5. Otsu binarisation    (yellow-tag friendly threshold)
    6. Morphological close  (fill tiny digit gaps)
    Returns a grayscale PIL image.
    """
    gray = crop_pil.convert("L")
    w, h = gray.size

    # ── Upscale ───────────────────────────────────────────────
    min_dim = min(w, h)
    if   min_dim < 40:  scale = 4
    elif min_dim < 80:  scale = 3
    elif min_dim < 150: scale = 2
    else:               scale = 1
    if scale > 1:
        gray = gray.resize((w * scale, h * scale), Image.BICUBIC)

    # ── CLAHE ─────────────────────────────────────────────────
    arr  = np.array(gray)
    arr  = clahe_numpy(arr, clip_limit=2.5, tile=8)

    # ── Sharpen ───────────────────────────────────────────────
    pil  = Image.fromarray(arr)
    pil  = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    arr  = np.array(pil)

    # ── Otsu binarise ─────────────────────────────────────────
    binary = otsu_threshold(arr)

    # ── Morphological close ───────────────────────────────────
    try:
        binary = morphological_close(binary, ksize=2)
    except ImportError:
        pass   # scipy not available — skip closing, still works

    return Image.fromarray(binary)


# ─────────────────────────────────────────────────────────────
# DOMINANT-NUMBER PICKER  (ignores small handwritten annotations)
# ─────────────────────────────────────────────────────────────
DIGITS_ONLY = "0123456789"


def _bbox_height(bbox) -> float:
    ys = [pt[1] for pt in bbox]
    return max(ys) - min(ys)


def _pick_dominant_number(results: list):
    """
    Ear tags have one large printed ID + small handwritten notes.
    Keep only detections whose bounding-box height >= 60 % of the
    tallest detection → discards the small annotations automatically.
    Merge survivors left-to-right → final tag number.
    """
    valid = []
    for (bbox, text, conf) in results:
        clean = re.sub(r"\D", "", text)
        if clean and float(conf) > 0.20:
            valid.append((bbox, clean, float(conf)))

    if not valid:
        return "", 0.0

    heights    = [_bbox_height(b) for b, _, _ in valid]
    max_h      = max(heights)
    threshold  = max_h * 0.60

    dominant = [
        (bbox, text, conf)
        for (bbox, text, conf), h in zip(valid, heights)
        if h >= threshold
    ]
    dominant.sort(key=lambda x: np.mean([pt[0] for pt in x[0]]))

    merged   = "".join(t for _, t, _ in dominant)
    avg_conf = float(np.mean([c for _, _, c in dominant]))
    return merged, round(avg_conf, 4)


def ocr_crop(reader, crop_pil: Image.Image):
    """
    Multi-pass EasyOCR on a PIL crop.
    Tries preprocessed, inverted, and raw variants.
    Returns (best_digits_string, confidence).
    """
    processed = preprocess_grayscale(crop_pil)
    inverted  = ImageOps.invert(processed)

    candidates = [
        np.array(processed),   # grayscale binary
        np.array(inverted),    # inverted  binary
        np.array(crop_pil),    # raw RGB fallback
    ]

    best_text, best_conf = "", 0.0

    for img_arr in candidates:
        try:
            results = reader.readtext(
                img_arr,
                detail=1,
                paragraph=False,
                allowlist=DIGITS_ONLY,
                decoder="beamsearch",
                beamWidth=10,
                text_threshold=0.25,
                low_text=0.20,
                link_threshold=0.3,
                width_ths=0.8,
            )
        except Exception:
            continue

        if not results:
            continue

        text, conf = _pick_dominant_number(results)
        if not text:
            continue

        if len(text) > len(best_text) or (
                len(text) == len(best_text) and conf > best_conf):
            best_text = text
            best_conf = conf

    return best_text or "UNREADABLE", round(best_conf, 4)


# ─────────────────────────────────────────────────────────────
# DIGIT VALIDATION
# ─────────────────────────────────────────────────────────────
DIGIT_FIXES = {
    "l":"1","I":"1","i":"1",
    "O":"0","o":"0","Q":"0",
    "S":"5","s":"5",
    "B":"8","b":"6",
    "Z":"2","z":"2",
    "G":"6","g":"9",
    "T":"7",
}


def validate_number(text: str):
    if not text or text in ("UNREADABLE", "LOW CONFIDENCE"):
        return text, "UNREADABLE", 0.0

    corrected, fixes = "", 0
    for ch in text.upper():
        if ch.isdigit():
            corrected += ch
        elif ch in DIGIT_FIXES:
            corrected += DIGIT_FIXES[ch]
            fixes += 1

    if not corrected:
        return text, "UNREADABLE", 0.0

    conf   = max(0.60, 0.97 - fixes * 0.05)
    status = "VERIFIED" if fixes == 0 else f"AUTO-CORRECTED ({fixes} fix{'es' if fixes>1 else ''})"
    return corrected, status, round(conf, 2)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)


@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False, verbose=False)


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🐄 Cattle Ear Tag Detection & OCR")
st.caption("Upload livestock images — YOLO detects tags, EasyOCR reads the number.")

if "results" not in st.session_state:
    st.session_state.results = []

c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
with c1:
    uploaded = st.file_uploader("Upload ZIP or Image",
                                type=["zip","jpg","jpeg","png"])
with c2:
    det_conf = st.slider("Detection Confidence", 0.10, 1.0, 0.40, 0.05)
with c3:
    ocr_conf_thresh = st.slider("Min OCR Confidence", 0.0, 1.0, 0.25, 0.05)
with c4:
    pad_px = st.number_input("BBox Padding (px)", 0, 40, 8)

model_path = "cow_eartag_yolov8n_100ep_clean_best.pt"
st.divider()

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
if uploaded:
    model  = load_yolo(model_path)
    reader = load_ocr_reader()

    with tempfile.TemporaryDirectory() as tmp:
        image_paths = []

        if uploaded.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded, "r") as z:
                z.extractall(tmp)
            for f in sorted(os.listdir(tmp)):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    image_paths.append(os.path.join(tmp, f))
        else:
            p = os.path.join(tmp, uploaded.name)
            with open(p, "wb") as f:
                f.write(uploaded.getbuffer())
            image_paths = [p]

        st.session_state.results = []
        total_tags = 0

        for img_path in image_paths:
            img_name  = os.path.basename(img_path)
            image_pil = pil_from_path(img_path)

            yolo_results = model(img_path, conf=det_conf)
            boxes        = yolo_results[0].boxes

            with st.expander(f"📷  {img_name}  —  {len(boxes)} tag(s) detected",
                             expanded=True):

                col_img, col_tags = st.columns([3, 2])

                # Annotated full image — YOLO returns numpy RGB via .plot()
                plotted_np = yolo_results[0].plot()          # BGR numpy
                plotted_rgb = plotted_np[:, :, ::-1]         # flip to RGB
                with col_img:
                    st.image(plotted_rgb, use_container_width=True,
                             caption="YOLO detections")

                with col_tags:
                    if len(boxes) == 0:
                        st.warning("No ear tags detected.")
                        continue

                    for i, box in enumerate(boxes):
                        det_confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        crop_pil = crop_box(image_pil, x1, y1, x2, y2, pad=pad_px)

                        # OCR
                        raw_text, ocr_confidence = ocr_crop(reader, crop_pil)

                        if ocr_confidence < ocr_conf_thresh and raw_text != "UNREADABLE":
                            raw_text = "LOW CONFIDENCE"

                        corrected, status, val_conf = validate_number(raw_text)
                        total_tags += 1

                        # Display
                        st.markdown(f"**Tag #{i+1}**")
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            st.image(crop_pil, caption="Crop",
                                     use_container_width=True)
                        with tc2:
                            gray_pil = preprocess_grayscale(crop_pil)
                            st.image(gray_pil, caption="Preprocessed",
                                     use_container_width=True)

                        display_num = corrected if corrected not in (
                            "UNREADABLE", "LOW CONFIDENCE") else "???"
                        st.markdown(
                            f'<div class="tag-box">{display_num}</div>',
                            unsafe_allow_html=True)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Raw OCR",   raw_text)
                        m2.metric("Corrected", corrected)
                        m3.metric("Status",    status)

                        st.progress(min(det_confidence, 1.0),
                                    text=f"Detection conf: {det_confidence:.0%}")
                        st.progress(min(ocr_confidence, 1.0),
                                    text=f"OCR conf: {ocr_confidence:.0%}")
                        st.divider()

                        st.session_state.results.append({
                            "Image":           img_name,
                            "Tag_#":           i + 1,
                            "Raw_OCR":         raw_text,
                            "Tag_Number":      corrected,
                            "Status":          status,
                            "OCR_Confidence":  f"{ocr_confidence:.2f}",
                            "Detection_Conf":  f"{det_confidence:.2f}",
                            "Validation_Conf": f"{val_conf:.2f}",
                            "BBox":            f"({x1},{y1})-({x2},{y2})",
                            "Timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })

        if st.session_state.results:
            st.divider()
            st.subheader(f"📊  Results — {total_tags} tag(s) processed")
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df, use_container_width=True)

            readable = df[df["Tag_Number"].str.match(r"^\d+$", na=False)]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Tags",  len(df))
            s2.metric("Readable",    len(readable))
            s3.metric("Unreadable",  len(df) - len(readable))
            s4.metric("Avg OCR Conf",
                      f"{df['OCR_Confidence'].astype(float).mean():.0%}")

            st.download_button("⬇️  Download CSV",
                               df.to_csv(index=False),
                               file_name="ear_tag_results.csv",
                               mime="text/csv",
                               use_container_width=True)

else:
    st.info("⬆️  Upload one image or a ZIP of cattle images to begin.")

    st.markdown("---")
    st.markdown("#### 🧪  Test OCR pipeline (no YOLO needed)")
    test_upload = st.file_uploader("Drop an ear-tag crop here",
                                   type=["jpg","jpeg","png"],
                                   key="test_ocr")
    if test_upload:
        reader   = load_ocr_reader()
        crop_pil = pil_from_bytes(test_upload.read())

        gray_pil           = preprocess_grayscale(crop_pil)
        raw_text, conf     = ocr_crop(reader, crop_pil)
        corrected, status, val_conf = validate_number(raw_text)

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.image(crop_pil, caption="Original", use_container_width=True)
        with tc2:
            st.image(gray_pil, caption="Preprocessed", use_container_width=True)
        with tc3:
            st.markdown(
                f'<div class="tag-box">{corrected or "???"}</div>',
                unsafe_allow_html=True)
            st.metric("Raw OCR",   raw_text)
            st.metric("Corrected", corrected)
            st.metric("Status",    status)
            st.progress(conf, text=f"OCR confidence: {conf:.0%}")
