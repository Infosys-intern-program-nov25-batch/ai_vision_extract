import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = 128
MODEL_PATH = "segmentation_model.h5"

st.set_page_config(page_title="Assignment 04 – Image Segmentation", layout="wide")
st.title("Assignment 04 – Image Segmentation")
st.caption("Background Removal using CNN and COCO 2017 Sample Dataset")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)

    # ---------------- PREPROCESS ----------------
    resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    resized = resized / 255.0
    input_img = np.expand_dims(resized, axis=0)

    # ---------------- PREDICT MASK ----------------
    pred_mask = model.predict(input_img)[0]

    # Convert to binary mask
    pred_mask = (pred_mask > 0.4).astype(np.uint8)

    # Resize mask to original image size
    mask_resized = cv2.resize(
        pred_mask,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert mask to 3 channels
    mask_3ch = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)

    # ---------------- APPLY MASK ----------------
    foreground = image_np * mask_3ch

    with col2:
        st.subheader("Foreground Extracted (Background Removed)")
        st.image(foreground, use_container_width=True)
