import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Image Background Removal",
    page_icon="🖼️",
    layout="wide"
)

MODEL_PATH = "segmentation_model.h5"
IMG_SIZE = 128

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_segmentation_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_segmentation_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f1117;
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
}
.sub-title {
    font-size: 20px;
    color: #b0b3b8;
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #4cc9f0;
    margin-top: 20px;
}
.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🖼️ Image Background Removal</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>CNN-based Image Segmentation using COCO 2017 Sample Dataset</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("Segmentation Threshold", 0.1, 0.9, 0.5)
show_mask = st.sidebar.checkbox("Show Segmentation Mask", True)
show_overlay = st.sidebar.checkbox("Show Mask Overlay", True)

st.sidebar.markdown("---")
st.sidebar.markdown("📘 **Instructions**")
st.sidebar.markdown("""
1. Upload an image  
2. Model predicts foreground mask  
3. Background is removed  
""")

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader(
    "📤 Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    # ---------------- SEGMENTATION ----------------
    resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE)) / 255.0
    pred = model.predict(resized[None, ...])[0, :, :, 0]
    mask = (pred > threshold).astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Foreground extraction
    foreground = img_np * np.repeat(mask[:, :, None], 3, axis=2)

    # Overlay
    overlay = img_np.copy()
    overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([0, 255, 255]) * 0.5

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>📷 Original Image</div>", unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)

        st.markdown("<div class='section-title'>🎯 Foreground Extracted</div>", unsafe_allow_html=True)
        st.image(foreground.astype(np.uint8), use_container_width=True)

    with col2:
        if show_mask:
            st.markdown("<div class='section-title'>🔵 Segmentation Mask</div>", unsafe_allow_html=True)
            st.image(mask * 255, use_container_width=True)

        if show_overlay:
            st.markdown("<div class='section-title'>🎨 Mask Overlay</div>", unsafe_allow_html=True)
            st.image(overlay.astype(np.uint8), use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    st.markdown("---")
    result_img = Image.fromarray(foreground.astype(np.uint8))
    st.download_button(
        "⬇️ Download Background Removed Image",
        data=result_img.tobytes(),
        file_name="foreground.png",
        mime="image/png"
    )

else:
    st.info("⬆️ Upload an image to begin background removal")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 | AI Vision Project<br>Made by <b>Priasha Patle</b></div>",
    unsafe_allow_html=True
)
