import os

# --- 1. NEW REQUIREMENTS (TensorFlow + YOLO) ---
requirements_txt = """tensorflow
ultralytics
streamlit
opencv-python
numpy
Pillow
pycocotools
matplotlib
"""

# --- 2. NEW TRAIN SCRIPT (Keras/TensorFlow) ---
train_py = """import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence

# --- CONFIG ---
DATA_DIR = 'data'  # Ensure your COCO data is here
IMG_DIR = os.path.join(DATA_DIR, 'val2017')
ANN_FILE = os.path.join(DATA_DIR, 'annotations/instances_val2017.json')
MODEL_DIR = 'model'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'segmentation_model.h5')
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10
# We train on these categories to match your dog/cat/person images
CLASSES = ['person', 'dog', 'cat']

class CocoKerasGenerator(Sequence):
    def __init__(self, img_dir, ann_file, batch_size, img_size, classes):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.cat_ids = self.coco.getCatIds(catNms=classes)
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        self.indexes = np.arange(len(self.img_ids))

    def __len__(self):
        return int(np.floor(len(self.img_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_img_ids = [self.img_ids[k] for k in indexes]
        
        X = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)

        for i, img_id in enumerate(batch_img_ids):
            # Load Image
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load Mask
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            mask = np.zeros((img_info['height'], img_info['width']))
            for ann in anns:
                mask = np.maximum(mask, self.coco.annToMask(ann))
            
            # Resize
            X[i] = cv2.resize(img, (self.img_size, self.img_size)) / 255.0
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            y[i] = mask_resized[:, :, np.newaxis]
            
        return X, y

def build_unet(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def train():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    print("Loading Data Generator...")
    try:
        train_gen = CocoKerasGenerator(IMG_DIR, ANN_FILE, BATCH_SIZE, IMG_SIZE, CLASSES)
    except Exception as e:
        print(f"Error: Could not load data. Check paths.\\n{e}")
        return

    print("Building Model...")
    model = build_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Starting Training (This may take a while)...")
    model.fit(train_gen, epochs=EPOCHS)
    
    model.save(MODEL_SAVE_PATH)
    print(f"SUCCESS: Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
"""

# --- 3. YOUR NEW APP (From your upload) ---
app_py = """import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEG_IMG_SIZE = 128
SEG_MODEL_PATH = "model/segmentation_model.h5"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # Load Keras Segmentation Model
    if not os.path.exists(SEG_MODEL_PATH):
        return None, None
    seg_model = load_model(SEG_MODEL_PATH)
    
    # Load YOLO Model (downloads automatically if missing)
    yolo_model = YOLO("yolov8n.pt") 
    return seg_model, yolo_model

try:
    seg_model, yolo_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    seg_model, yolo_model = None, None

# ---------------- UI ----------------
st.set_page_config(page_title="Project", layout="wide")
st.markdown(\"\"\"
<style>
h1 { color: #ff4b4b; font-size: 48px; font-weight: bold; }
h2 { color: #ff9900; }
h3 { color: #00cc99; }
</style>
\"\"\", unsafe_allow_html=True)

st.title("🖼️ Object Detection + Segmentation App")
st.markdown("#### Using **YOLOv8** for Detection + **CNN U-Net** for Segmentation")

if seg_model is None:
    st.error("Model file not found! Please run 'python train.py' first.")
else:
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
                if hasattr(yolo_model, 'names'):
                    label = yolo_model.names[cls_id]
                else:
                    label = str(cls_id)
                detected_objects.append((label, conf))

                cv2.rectangle(det_img, (x1,y1),(x2,y2),(255,102,0),3)
                cv2.putText(det_img,f"{label} {conf:.2f}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,102,0),2)

        # ---------------- Segmentation ----------------
        # Resize for model
        seg_input = cv2.resize(img_np,(SEG_IMG_SIZE,SEG_IMG_SIZE)).astype(np.float32)/255.0
        seg_input = np.expand_dims(seg_input, axis=0)
        
        # Predict
        pred = seg_model.predict(seg_input)[0,:,:,0]
        mask = (pred>0.5).astype(np.uint8)
        
        # Post-process mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.resize(mask,(w,h), interpolation=cv2.INTER_NEAREST)

        # ---------------- Overlay ----------------
        overlay = img_np.copy()
        color = np.array([0,255,255], dtype=np.uint8)  # Cyan
        overlay[mask==1] = overlay[mask==1]*0.5 + color*0.5
        blended = cv2.addWeighted(img_np,0.7,overlay,0.3,0)
        
        # Segmented Image (Cutout)
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
"""

readme_md = """# Assignment 04: YOLO Detection + Keras Segmentation

## Setup
1. **Data:** Ensure `data/val2017` and `data/annotations` exist (same as before).
2. **Install:** `pip install -r requirements.txt`
3. **Train:** `python train.py` (Creates model/segmentation_model.h5)
4. **Run:** `streamlit run app.py`
"""

# --- WRITE FILES ---
files = {
    "requirements.txt": requirements_txt,
    "train.py": train_py,
    "app.py": app_py,
    "README.md": readme_md
}

# Create model directory
os.makedirs("model", exist_ok=True)

for name, content in files.items():
    with open(name, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated: {name}")

print("\\nSuccess! Project converted to TensorFlow/YOLO.")
print("Run 'python train.py' to generate the .h5 model file.")