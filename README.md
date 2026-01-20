# AI Vision Extract
# VisionExtract AI: Hybrid Cloud-Local Image Segmentation 🖼️✂️

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%2F%20JS-E34F26)
![Model](https://img.shields.io/badge/Model-DeepLabV3%2B-blue)
![Deployment](https://img.shields.io/badge/Architecture-Hybrid%20Cloud-blueviolet)

## 🎯 Project Overview

**VisionExtract AI** is a full-stack deep learning application that instantly removes image backgrounds with high precision.

Unlike traditional web apps that rely on expensive cloud GPUs, this project utilizes a **Hybrid Architecture**:
1.  **Frontend (The Face):** A lightweight, responsive UI hosted on the cloud (Vercel).
2.  **Backend (The Brain):** A local high-performance FastAPI server running the heavy AI model.
3.  **The Bridge:** Secure tunneling via **ngrok** connects the two in real-time.

Built with **DeepLabV3+ (ResNet101)** trained on the COCO 2017 dataset, achieving **67.4% mIoU**.

---

## 🏗️ Technical Architecture

This project solves the "Heavy Model Deployment" problem by splitting the stack:

```mermaid
graph LR
    User["User Device"] -- HTTPS --> Vercel["Frontend (Vercel)"]
    Vercel -- "API Request" --> Ngrok["Ngrok Tunnel"]
    Ngrok -- "Secure Tunnel" --> Localhost["Local Machine"]
    Localhost -- Uvicorn --> FastAPI["FastAPI Server"]
    FastAPI -- Inference --> PyTorch["DeepLabV3 Model"]
🚀 Key Features🧠 Intelligent BackendState-of-the-Art Model: Uses DeepLabV3_ResNet101 for superior edge detection.Smart Resizing: Implements Lanczos resampling to automatically resize 4K+ images to 1024px, increasing inference speed by 3x without visible quality loss.FastAPI & Uvicorn: Asynchronous request handling for non-blocking performance.🎨 Reactive FrontendCanvas "Baking": Download logic uses HTML5 Canvas to merge the subject, background color, and visual filters (Brightness/Contrast) into a single high-quality PNG.Real-Time Enhancements: Adjust Brightness and Contrast instantly.Hybrid Connectivity: Dynamically connects to the local backend via public ngrok URLs.📊 Performance MetricsModelBackboneMean IoUPixel AccuracyDeepLabV3+ResNet10167.4%92.4%Dataset: Trained on 118K images from COCO 2017.🛠️ Tech StackModel: PyTorch, Torchvision, DeepLabV3+Backend: Python 3.9, FastAPI, Uvicorn, Pillow (PIL)Frontend: HTML5, CSS3, Vanilla JavaScriptTunneling: Ngrok (Cross-network exposure)Hosting: Vercel (Frontend assets)💻 Installation & Setup (Run it Locally)PrerequisitesPython 3.8+ installed.Ngrok installed and authenticated.Step 1: Clone RepositoryBashgit clone [https://github.com/yourusername/vision-extract-ai.git](https://github.com/yourusername/vision-extract-ai.git)
cd vision-extract-ai
Step 2: Setup Backend ("The Brain")Navigate to the project folder and install dependencies:Bashpip install -r requirements.txt
Start the local server:Bashpython backend.py
You should see: INFO: Uvicorn running on http://127.0.0.1:8000Step 3: Start the TunnelOpen a new terminal and expose your local port 8000:Bashngrok http 8000
Copy the Forwarding URL (e.g., https://random-name.ngrok-free.dev)Step 4: Connect FrontendOpen frontend/script.js.Find the fetch URL line.Replace it with your new Ngrok URL:JavaScriptconst response = await fetch('[https://your-ngrok-url.ngrok-free.dev/remove-bg/](https://your-ngrok-url.ngrok-free.dev/remove-bg/)', ...);
Open frontend/index.html in your browser.📂 Project Structurevision-extract-ai/
├── backend.py                # FastAPI Server & Model Inference Logic
├── requirements.txt          # Python Dependencies
├── frontend/
│   ├── index.html            # Main UI
│   ├── style.css             # Dark Mode & Responsive Design
│   └── script.js             # Canvas Logic & API Communication
├── models/
│   └── deeplabv3_resnet101.pth  # (Optional: Auto-downloads on first run)
└── README.md
🗺️ Development RoadmapPhaseMilestoneStatus1Data Pipeline (COCO 2017 cleaning & prep)✅ Completed2Model Training (DeepLabV3 optimization)✅ Completed3Backend Dev (FastAPI Integration)✅ Completed4Hybrid Deployment (Ngrok + Vercel Setup)✅ Completed5Frontend Polish (Canvas Baking & UI)✅ Completed📄 LicenseMIT License - see LICENSE file.⭐ Star this repo if you find the Hybrid Architecture approach useful!Built by Infosys Springboard Data Science Intern Team
