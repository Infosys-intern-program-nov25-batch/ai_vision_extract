import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import io


# U-NET MODEL 
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))



# LOAD MODEL (Cached)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load('model.pth', map_location=device)
        
        # Check if it's a checkpoint dict or just weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, device, None
    except Exception as e:
        return None, device, str(e)


# PREPROCESSING

def preprocess_image(image, size=256):
    """Prepare image for model"""
    original_size = image.size
    img = image.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, original_size

# POSTPROCESSING

def postprocess_mask(mask_tensor, original_size):
    """Convert model output to mask"""
    mask = mask_tensor.squeeze().cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, original_size)
    return mask

def remove_background(image, mask, bg_option):
    # sourcery skip: extract-duplicate-method
    """Apply mask and create output based on selected background"""
    img_array = np.array(image)
    mask_normalized = mask.astype(np.float32) / 255.0
    
    if bg_option == "Transparent PNG":
        # Create RGBA
        rgba = np.dstack([img_array, mask])
        return Image.fromarray(rgba, 'RGBA')
    
    elif bg_option == "White Background":
        white_bg = np.ones_like(img_array) * 255
        mask_3ch = np.stack([mask_normalized] * 3, axis=2)
        result = (img_array * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        return Image.fromarray(result)
    
    elif bg_option == "Black Background":
        mask_3ch = np.stack([mask_normalized] * 3, axis=2)
        result = (img_array * mask_3ch).astype(np.uint8)
        return Image.fromarray(result)
    
    elif bg_option == "Blurred Background":
        blurred = cv2.GaussianBlur(img_array, (41, 41), 0)
        mask_3ch = np.stack([mask_normalized] * 3, axis=2)
        result = (img_array * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        return Image.fromarray(result)


# STREAMLIT APP INTERFACE

def main():
    st.set_page_config(
        page_title="Image Segmentation & Background Removal",
        page_icon="🎨",
        layout="wide"
    )
    # Add title for the website 
    st.set_page_config(page_title="AI Vision Extract", layout="wide", page_icon="🎯")
    # Header
    st.title("🎨 Image Segmentation & Background Removal")
    #st.markdown("### CNN-based segmentation using COCO trained model")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, device, error = load_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Make sure 'model.pth' is in the same folder as app.py")
        return
    
    st.success(f"✅ Model loaded successfully! Using: {device}")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    bg_option = st.sidebar.selectbox(
        "Background Style:",
        ["Transparent PNG", "White Background", "Black Background", "Blurred Background"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 How to Use:")
    st.sidebar.markdown("1. Upload an image")
    st.sidebar.markdown("2. Wait for segmentation")
    st.sidebar.markdown("3. Choose background style")
    st.sidebar.markdown("4. Download result")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🚀 Segment & Remove Background", type="primary"):
            with st.spinner("Processing..."):
                # Preprocess
                img_tensor, original_size = preprocess_image(image)
                img_tensor = img_tensor.to(device)
                
                # Predict
                with torch.no_grad():
                    mask_tensor = model(img_tensor)
                
                # Postprocess
                mask = postprocess_mask(mask_tensor, original_size)
                
                # Apply background removal
                result = remove_background(image, mask, bg_option)
                
                # Display results
                with col2:
                    st.subheader("🎭 Predicted Mask")
                    st.image(mask, use_container_width=True, clamp=True)
                
                with col3:
                    st.subheader("✨ Final Result")
                    st.image(result, use_container_width=True)
                
                # Download button
                st.markdown("---")
                buf = io.BytesIO()
                if bg_option == "Transparent PNG":
                    result.save(buf, format='PNG')
                    file_ext = "png"
                else:
                    result.save(buf, format='JPEG')
                    file_ext = "jpg"
                
                buf.seek(0)
                
                st.download_button(
                    label="⬇️ Download Result",
                    data=buf,
                    file_name=f"segmented_result.{file_ext}",
                    mime=f"image/{file_ext}",
                    type="primary"
                )
                
                st.success("✅ Processing complete!")
    
    else:
        st.info("👆 Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with:** PyTorch + Streamlit | **Model:** U-Net CNN | **Dataset:** COCO 2017")

if __name__ == "__main__":
    main()
