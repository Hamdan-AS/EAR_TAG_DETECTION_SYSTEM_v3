import streamlit as st
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import re
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cow Tag Scanner", layout="wide")
st.title("🐄 Cow Ear Tag AI")
st.write("Detect ear tags and read ID numbers using YOLOv8 & EasyOCR. Upload a single image **or a ZIP** of images.")

MISHAP_MAP = {
    "|": "1", "I": "1", "l": "1", "[": "1", "]": "1", "(": "1", ")": "1",
    "O": "0", "o": "0", "S": "5", "s": "5", "B": "8", "G": "6"
}

@st.cache_resource
def load_models():
    model  = YOLO("cow_eartag_yolov8n_100ep_clean_best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, ocr_reader = load_models()

# ─────────────────────────────────────────────────────────────
# 2. SIDEBAR SETTINGS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
conf_level = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)
padding    = st.sidebar.number_input("Crop Padding (px)", 0, 100, 20)

# ─────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────

def _bbox_height(bbox) -> float:
    """Return pixel height of an EasyOCR bounding box [[x,y],...]."""
    ys = [pt[1] for pt in bbox]
    return max(ys) - min(ys)


def pick_dominant_number(ocr_results: list) -> tuple[str, str]:
    """
    Discard small handwritten annotations; keep only the tallest
    (= largest printed) text regions, merged left-to-right.
    Returns (raw_merged_text, digits_only).
    """
    if not ocr_results:
        return "", ""

    heights   = [_bbox_height(r[0]) for r in ocr_results]
    max_h     = max(heights)
    threshold = max_h * 0.60        # ignore anything < 60 % of tallest

    dominant = [
        r for r, h in zip(ocr_results, heights) if h >= threshold
    ]
    # sort left-to-right by x-centre of bbox
    dominant.sort(key=lambda r: np.mean([pt[0] for pt in r[0]]))

    raw_text = "".join(r[1] for r in dominant).strip()

    # apply mishap fixes → digits only
    final_id = ""
    for ch in raw_text:
        if ch.isdigit():
            final_id += ch
        elif ch in MISHAP_MAP:
            final_id += MISHAP_MAP[ch]

    return raw_text, final_id


def preprocess_crop(crop_pil: Image.Image) -> Image.Image:
    """Grayscale → contrast boost → auto-level."""
    img = ImageOps.grayscale(crop_pil)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageOps.autocontrast(img)
    return img


def safe_crop(img: Image.Image, x1, y1, x2, y2, pad: int) -> Image.Image:
    """Crop with padding, clamped to image bounds."""
    w, h = img.size
    return img.crop((
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ))


def load_images_from_upload(uploaded) -> list[tuple[str, Image.Image]]:
    """
    Accept a single image OR a ZIP file.
    Returns list of (filename, PIL.Image).
    """
    images = []

    if uploaded.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(uploaded.read()), "r") as zf:
            entries = sorted(zf.namelist())
            for entry in entries:
                # skip directories and hidden / macOS metadata files
                if entry.endswith("/") or os.path.basename(entry).startswith("__"):
                    continue
                ext = os.path.splitext(entry)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                try:
                    data = zf.read(entry)
                    pil  = Image.open(io.BytesIO(data)).convert("RGB")
                    images.append((os.path.basename(entry), pil))
                except Exception:
                    pass   # skip unreadable files silently
    else:
        pil = Image.open(uploaded).convert("RGB")
        images.append((uploaded.name, pil))

    return images


def process_image(img_name: str,
                  img: Image.Image,
                  conf: float,
                  pad: int) -> tuple[np.ndarray, list[dict]]:
    """
    Run YOLO + OCR on one PIL image.
    Returns (annotated_rgb_array, list_of_result_dicts).
    """
    results   = detector(img, conf=conf)[0]
    annotated = results.plot()[:, :, ::-1]   # BGR → RGB
    boxes     = results.boxes
    records   = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2    = map(int, box.xyxy[0])
        det_conf           = float(box.conf[0])
        tag_crop           = safe_crop(img, x1, y1, x2, y2, pad)
        clean_crop         = preprocess_crop(tag_crop)
        crop_np            = np.array(clean_crop)
        ocr_out            = ocr_reader.readtext(crop_np)
        raw_text, final_id = pick_dominant_number(ocr_out)

        records.append({
            "Image":          img_name,
            "Tag_#":          i + 1,
            "Tag_Number":     final_id  if final_id  else "UNREADABLE",
            "Raw_OCR":        raw_text  if raw_text  else "",
            "Detection_Conf": f"{det_conf:.2f}",
            "BBox":           f"({x1},{y1})-({x2},{y2})",
            "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # keep PIL crops for display (not in final CSV)
            "_tag_crop":      tag_crop,
            "_clean_crop":    clean_crop,
        })

    return annotated, records


# ─────────────────────────────────────────────────────────────
# 4. FILE UPLOADER  (image OR zip)
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a cattle image or ZIP of images…",
    type=["zip","jpg", "jpeg", "png"],
)

# ─────────────────────────────────────────────────────────────
# 5. MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────
if uploaded_file:

    all_images = load_images_from_upload(uploaded_file)

    if not all_images:
        st.error("No valid images found in the uploaded file.")
        st.stop()

    st.info(f"Processing **{len(all_images)}** image(s) …")

    all_records = []   # accumulate across all images for CSV

    for img_name, img in all_images:

        annotated, records = process_image(img_name, img, conf_level, padding)
        all_records.extend(records)

        with st.expander(f"📷  {img_name}  —  {len(records)} tag(s)", expanded=True):

            st.image(annotated, caption="YOLO detections",
                     use_container_width=True)

            if not records:
                st.info("No tags detected. Try lowering the confidence slider.")
                continue

            st.subheader(f"Found {len(records)} tag(s):")

            for rec in records:
                cols = st.columns([1, 2])

                with cols[0]:
                    st.image(rec["_tag_crop"],
                             caption=f"Tag #{rec['Tag_#']} — raw crop")
                    st.image(rec["_clean_crop"],
                             caption="Preprocessed (grayscale)")

                with cols[1]:
                    num = rec["Tag_Number"]
                    if num and num != "UNREADABLE":
                        st.markdown(f"""
                        <div style="background:#1e293b;color:#facc15;
                                    font-size:2rem;font-weight:900;
                                    letter-spacing:6px;text-align:center;
                                    padding:12px;border-radius:8px;
                                    font-family:monospace;">
                            {num}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("Could not read numbers clearly.")

                    st.text(f"Raw OCR : {rec['Raw_OCR']}")
                    st.text(f"Det conf: {rec['Detection_Conf']}")

                st.divider()

    # ─────────────────────────────────────────────────────────
    # 6. SUMMARY TABLE + CSV DOWNLOAD
    # ─────────────────────────────────────────────────────────
    if all_records:
        st.divider()
        st.subheader("📊 All Results")

        # Drop internal PIL columns before showing / exporting
        export_cols = ["Image","Tag_#","Tag_Number","Raw_OCR",
                       "Detection_Conf","BBox","Timestamp"]
        df = pd.DataFrame(all_records)[export_cols]
        st.dataframe(df, use_container_width=True)

        readable    = df[df["Tag_Number"].str.match(r"^\d+$", na=False)]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tags",  len(df))
        m2.metric("Readable",    len(readable))
        m3.metric("Unreadable",  len(df) - len(readable))

        st.download_button(
            "⬇️  Download CSV",
            df.to_csv(index=False),
            file_name="ear_tag_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("Please upload an image (JPG/PNG) or a ZIP of images to begin.")

# ─────────────────────────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit, YOLOv8, and EasyOCR")
