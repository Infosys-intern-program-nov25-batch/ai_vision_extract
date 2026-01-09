import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
from ultralytics import YOLO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEG_IMG_SIZE = 128
SEG_MODEL_PATH = "model/segmentation_model.h5"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    seg_model = load_model(SEG_MODEL_PATH)
    yolo_model = YOLO("yolov8n.pt")
    return seg_model, yolo_model

seg_model, yolo_model = load_models()

# ---------------- UI ----------------
st.set_page_config(page_title="Project", layout="wide")
st.markdown("""
<style>
h1 {
    color: #ff4b4b;
    font-size: 48px;
    font-weight: bold;
}
h2 {
    color: #ff9900;
}
h3 {
    color: #00cc99;
}
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Object Detection + Segmentation App")
st.markdown("#### Using **YOLOv8** for Detection + **CNN U-Net** for Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    # ---------------- YOLO Detection ----------------
    det_img = img_np.copy()
    results = yolo_model(det_img)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls_id]
            detected_objects.append((label, conf))

            cv2.rectangle(det_img, (x1,y1),(x2,y2),(255,102,0),3)  # bright orange boxes
            cv2.putText(det_img,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,102,0),2)

    # ---------------- Segmentation ----------------
    seg_input = cv2.resize(img_np,(SEG_IMG_SIZE,SEG_IMG_SIZE)).astype(np.float32)/255.0
    seg_input = np.expand_dims(seg_input, axis=0)
    pred = seg_model.predict(seg_input)[0,:,:,0]
    mask = (pred>0.5).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.resize(mask,(w,h), interpolation=cv2.INTER_NEAREST)

    # ---------------- Overlay with colorful mask ----------------
    overlay = img_np.copy()
    color = np.array([0,255,255], dtype=np.uint8)  # bright cyan
    overlay[mask==1] = overlay[mask==1]*0.5 + color*0.5
    blended = cv2.addWeighted(img_np,0.7,overlay,0.3,0)
    segmented = img_np * np.repeat(mask[:,:,None],3,axis=2)

    # ---------------- Display ----------------
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📷 Original Image")
        st.image(img_np, use_column_width=True)

        st.markdown("## 🎯 YOLO Detection")
        st.image(det_img, use_column_width=True)

        if detected_objects:
            st.markdown("## 📌 Detected Objects")
            det_table = {f"Object {i+1}": [label, f"{conf:.2f}"] for i, (label, conf) in enumerate(detected_objects)}
            st.table(det_table)
        else:
            st.warning("No objects detected")

    with col2:
        st.markdown("## 🟢 Segmentation Mask")
        st.image(mask*255, use_column_width=True)

        st.markdown("## 🧩 Mask Overlay")
        st.image(blended, use_column_width=True)

        st.markdown("## ✂️ Segmented Image")
        st.image(segmented.astype(np.uint8), use_column_width=True)

    st.markdown("---")
    st.markdown("Made by 💛 **ACHAL PANDE**")
