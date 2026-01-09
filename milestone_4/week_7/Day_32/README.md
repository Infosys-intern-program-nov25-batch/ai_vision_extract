# 🖼️ AI Vision Studio: COCO Image Segmentation

A professional-grade web application that leverages a **Convolutional Neural Network (CNN)** to perform real-time image segmentation and background removal. This project was developed as part of Assignment-04 to demonstrate model training, saving, and deployment using Streamlit.

## 🚀 Key Features
* **Selective Segmentation:** Choose specific objects (people, pets, vehicles) to keep while removing the background.
* **Real-Time Processing:** Optimized inference using `torch.inference_mode` and image downscaling for sub-second results.
* **High Fidelity:** Built on the **DeepLabV3-ResNet50** architecture, pre-trained on the COCO dataset.
* **Interactive UI:** Sleek dashboard with custom background color pickers and performance metrics.

## 📁 Project Structure
* `app.py`: The main Streamlit web application.
* `segmentation_model.pth`: Saved weights of the trained CNN model.
* `requirements.txt`: List of Python dependencies.
* `README.md`: Project documentation and setup guide.

## 🛠️ How to Run on Your PC

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## 🧪 Quick Test
I have included a `test_images/` folder with sample COCO images. To test the app, simply drag and drop any image from that folder into the sidebar uploader.