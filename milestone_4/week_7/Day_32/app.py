import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# Mapping the 21 classes (Background + 20 Objects)
COCO_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

st.set_page_config(page_title="Selective Background Remover", layout="wide")
st.title("🎯 Selective Object Extractor")

@st.cache_resource
def load_model():
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.eval()
    return model

model = load_model()

# Sidebar for controls
st.sidebar.header("Settings")
target_objects = st.sidebar.multiselect(
    "Select objects to KEEP:", 
    COCO_CLASSES[1:], # Exclude background
    default=['person']
)

uploaded_file = st.sidebar.file_uploader("Upload a COCO image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    # Image Processing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Get the class index for each pixel
    output_predictions = output.argmax(0).cpu().numpy()
    
    # Create a mask that only keeps selected labels
    # Get the indices of the selected classes
    target_indices = [COCO_CLASSES.index(obj) for obj in target_objects]
    
    # Create binary mask: True if pixel class is in our target list
    final_mask = np.isin(output_predictions, target_indices).astype(np.uint8) * 255
    
    # Create RGBA Result
    img_np = np.array(img)
    res = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
    res[:, :, 3] = cv2.resize(final_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    with col2:
        st.subheader("Filtered Foreground")
        st.image(res, use_container_width=True)
        
        # Download button for the result
        result_img = Image.fromarray(res)
        st.download_button("Download Transparent PNG", data=uploaded_file, file_name="filtered.png", mime="image/png")
