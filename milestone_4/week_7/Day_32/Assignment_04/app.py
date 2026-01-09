import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io

# ---------------- MODEL ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = torch.cat([d, e1], dim=1)
        d = self.dec(d)
        return torch.sigmoid(self.out(d))

@st.cache_resource
def load_model():
    model = SimpleUNet()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Background Removal", layout="centered")
st.title("🖼️ Image Segmentation & Background Removal")
st.caption("CNN-based segmentation trained on COCO 2017")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Remove Background"):
        img = image.resize((128, 128))
        img = np.array(img) / 255.0
        img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            prob_mask = model(img)[0][0].numpy()

        # resize probability mask smoothly
        prob_mask = cv2.resize(prob_mask, image.size, interpolation=cv2.INTER_LINEAR)

        # show probability mask
        st.image((prob_mask * 255).astype(np.uint8),
             caption="Predicted Mask",
             use_container_width=True)

        # apply soft mask
        result = (np.array(image) * prob_mask[:, :, None]).astype(np.uint8)

        st.image(result,
             caption="Foreground Image",
             use_container_width=True)


        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), "result.png")
