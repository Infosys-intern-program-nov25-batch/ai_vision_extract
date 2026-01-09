import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Config
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Assignment 04", layout="wide")
st.title("Assignment-04: COCO Image Segmentation")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('coco_unet_model.h5')

model = load_model()

if not model:
    st.error("Model not found. Please train it in the notebook first.")
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        
        # Process Input
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize(IMG_SIZE)
        img_array = np.array(resized_image) / 255.0
        img_input = np.expand_dims(img_array, axis=0)
        
        with col1:
            st.image(original_image, caption="Original", use_container_width=True)

        if st.button("Remove Background"):
            # Predict
            pred = model.predict(img_input)
            mask = np.squeeze(pred)
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Resize mask to match original image
            mask_img = Image.fromarray(mask, mode='L')
            mask_img = mask_img.resize(original_image.size, Image.BILINEAR)
            
            # Apply Mask
            result = original_image.copy()
            result.putalpha(mask_img)
            
            with col2:
                st.image(result, caption="Background Removed", use_container_width=True)
