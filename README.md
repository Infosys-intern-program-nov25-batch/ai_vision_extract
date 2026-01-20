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
