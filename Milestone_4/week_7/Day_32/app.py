import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 128
MODEL_PATH = "model/segmentation_model.h5"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

st.set_page_config(page_title="CNN Background Removal", layout="centered")

st.title("🖼️ Image Background Removal")
st.write("CNN-based Image Segmentation using COCO 2017 Sample Dataset")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image_np, use_container_width=True)

    resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE)) / 255.0
    input_img = np.expand_dims(resized, axis=0)

    mask = model.predict(input_img)[0]
    mask = (mask > 0.5).astype(np.uint8)

    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    mask = np.expand_dims(mask, axis=-1)

    result = image_np * mask

    st.subheader("Foreground Extracted (Background Removed)")
    st.image(result, use_container_width=True)
