# 🖼️ AI Vision Extract — Subject Isolation via Deep Learning

**AI Vision Extract** automatically detects and extracts the main subject from any image, producing clean cutouts with customizable backgrounds.
Built using **DeepLabV3 with a ResNet101 backbone** and trained on the **COCO 2017 dataset**, the system delivers high-quality semantic segmentation suitable for real-world applications.

---

## 🎯 Project Overview

* **Model:** DeepLabV3 + ResNet101
* **Dataset:** COCO 2017 (122,206 filtered masks)
* **Mean IoU:** **67.4%**
* **Pixel Accuracy:** **92.4%**
* **Deployment:** Streamlit Web Application

The pipeline handles preprocessing, inference, auto-cropping, background replacement, and batch exports with a user-friendly interface.

---

## 🚀 Use Cases

* 📸 Photography automation
* 🛒 E-commerce product cutouts
* 🥽 AR / VR content pipelines
* 🎥 Virtual conferencing backgrounds

---

## ⚡ Quick Start (5 Minutes)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ai-vision-extract.git
cd ai-vision-extract/Project
```

### 2️⃣ Download Dataset

Download **COCO 2017 Dataset** and extract it to:

```text
data/coco2017/
├── train2017/
├── val2017/
└── annotations/
```

### 3️⃣ Setup Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 4️⃣ Download Trained Model

Download the pretrained model:

* **Model:** DeepLabV3-ResNet101 (COCO)

Move it to:

```bash
ai-vision-extract/Project/model/
```

### 5️⃣ Launch Web App 🚀

```bash
streamlit run app.py
```

Open 👉 **[http://localhost:8501](http://localhost:8501)**
Upload images → Instant subject extraction!

---

## 📊 Performance Metrics

| Model               | Mean IoU | Pixel Accuracy |
| ------------------- | -------- | -------------- |
| DeepLabV3-ResNet101 | 67.4%    | 92.4%          |

**Dataset Quality:**

* 123K+ images processed
* **99% valid masks retained after filtering**

---

## 🛠️ Technical Stack

* **Dataset:** COCO 2017 (118K train + 5K val images)
* **Architecture:** DeepLabV3 + ResNet101 (81 semantic classes)
* **Training:**

  * Optimizer: SGD
  * Learning Rate: 0.01
  * Loss: CrossEntropyLoss
  * Batch Size: 8
* **Preprocessing:** Anomaly detection, semantic mask extraction
* **Deployment:** Streamlit (GPU-optimized inference)

---

## ✨ Features

✅ Batch Upload (JPG / PNG / JPEG)
✅ Background Options (Black / White / Studio Gray)
✅ Brightness & Contrast Controls
✅ Auto-Crop to Subject Bounding Box
✅ Subject Coverage Metrics (%)
✅ Individual PNG Export + ZIP Download
✅ Dark Mode UI with Live Previews

---

## 📁 Project Structure

```text
ai-vision-extract/Project
├── app.py                     # Streamlit web interface
├── data/
│   └── test.jpg
├── model/
│   └── deeplabv3_resnet101_coco.pth
├── requirements.txt
```

---

## 🗺️ Development Roadmap

| Week | Milestone      | Deliverables                 |
| ---: | -------------- | ---------------------------- |
|  1–2 | Data Pipeline  | Clean masks, anomaly reports |
|  3–4 | Model Training | DeepLabV3 (67.4% mIoU)       |
|  5–6 | Inference      | Optimized model loading      |
|    7 | Web UI         | Streamlit batch app          |
|    8 | Documentation  | Technical report & demo      |

---

## 📈 Results Highlights

* **Data Quality:** 99% retention after aggressive filtering
* **Model Performance:** Industry-competitive segmentation accuracy
* **Production Ready:** Batch processing + GPU inference
* **UX Focused:** Fast feedback, intuitive controls

---

## 📚 Evaluation Metrics

* **Primary:** Mean Intersection over Union (mIoU)
* **Secondary:** Pixel Accuracy
* **Visual:** Before/After subject isolation quality

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## ⭐ Support

If this project helps your computer vision work, **please give it a star!**
Contributions, issues, and feature requests are welcome 🚀

---
