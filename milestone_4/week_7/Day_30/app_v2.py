import streamlit as st
import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image, ImageEnhance
import numpy as np
import io
import zipfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision Extract", layout="wide", page_icon="🎯")

# --- CUSTOM DARK MODE & UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main-header {
        font-size: 3rem; font-weight: 800;
        background: -webkit-linear-gradient(#FF4B4B, #FF8383);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; }
    .stButton>button { border-radius: 8px; background-color: #FF4B4B; color: white; border: none; font-weight: bold; width: 100%; }
    .stMetric { background-color: #1c2128; padding: 10px; border-radius: 10px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model_local():
    model_path = 'model/deeplabv3_resnet101_coco.pth'
    model = deeplabv3_resnet101(weights=None, weights_backbone=None, aux_classifier=True)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        st.error(f"❌ Model weights not found.")
        st.stop()

model = load_model_local()

# --- IMAGE DISPLAY UTILITY (SILENCES WARNINGS) ---
def safe_image(img, caption):
    """Displays image using the correct parameter for your Streamlit version."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

# --- PROCESSING HELPERS ---
def crop_to_subject(image, mask):
    coords = np.argwhere(mask)
    if coords.size == 0: return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return image.crop((x0, y0, x1, y1))
