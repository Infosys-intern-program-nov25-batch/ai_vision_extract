import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Image Background Removal",
    layout="centered"
)

st.title("🖼️ Image Background Removal")
st.write("CNN-based Image Segmentation using COCO 2017 sample dataset")

# ----------------- LOAD MODEL -----------------
model = load_model("cnn_segmentation.h5")

# ----------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "png"]
)

if uploaded_file is not None:

    # ---------- READ IMAGE ----------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    # ---------- PREPROCESS ----------
    img_small = cv2.resize(image, (128, 128))
    img_norm = img_small / 255.0

    # ---------- PREDICT ----------
    pred_mask = model.predict(img_norm[np.newaxis, ...])[0]

    # ---------- SAFE THRESHOLD ----------
    # Lower threshold improves weak CNN detection
    mask_small = (pred_mask > 0.3).astype(np.uint8)

    # ---------- SHOW MASK ----------
    st.subheader("Predicted Mask")
    st.image(mask_small * 255, clamp=True)

    # ---------- RESIZE MASK ----------
    mask_full = cv2.resize(
        mask_small,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # ---------- MORPHOLOGICAL CLEANING ----------
    kernel = np.ones((7, 7), np.uint8)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)

    # ---------- FAIL-SAFE ----------
    # If mask is almost empty, fallback to weak mask
    if np.sum(mask_full) < 500:
        st.warning("⚠ Weak segmentation detected. Applying fallback mask.")
        mask_full = cv2.dilate(mask_full, kernel, iterations=2)

    # ---------- APPLY MASK ----------
    mask_3c = np.repeat(mask_full[:, :, np.newaxis], 3, axis=2)
    foreground = image * mask_3c

    st.subheader("Foreground Image (Background Removed)")
    st.image(foreground, channels="BGR")
