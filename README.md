<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# AI Vision Extract: Subject Isolation via Deep Learning 🖼️✂️

## Project Overview 🎯

**AI Vision Extract** automatically detects and extracts the main subject from any image, producing clean cutouts with customizable backgrounds.  Built with DeepLabV3 ResNet101 and COCO 2017 dataset, it achieves **67.4% mIoU** and **92.4% pixel accuracy**.[^1][^2]

**Use Cases:** Photography automation, e-commerce, AR/VR, virtual conferencing backgrounds.[^1]

***

## 🚀 Quick Start (5 Minutes)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-vision-extract.git
cd ai-vision-extract
```


### 2. Download Dataset

- [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)[^1]
- Extract to `data/coco2017/` (train2017, val2017, annotations)


### 3. Setup Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```


### 4. Run Preprocessing \& Training

```bash
# Generate clean masks (runs anomaly detection + extraction)
python preprocess.py

# Train model
python train.py
```


### 5. Launch Web App 🚀

```bash
streamlit run app.py
```

Open `http://localhost:8501` → Upload images → Instant subject extraction!

***

## 📊 Key Performance Metrics

| Model | Mean IoU | Pixel Accuracy |
| :-- | :-- | :-- |
| **DeepLabV3-ResNet101** | **67.4%** | **92.4%** |

**Dataset:** 122,206 clean masks from COCO 2017 (99% valid after filtering)[^3]

***

## 🛠️ Technical Stack

```
Dataset: COCO 2017 (118K train + 5K val images)
Model: DeepLabV3 + ResNet101 backbone (81 output classes)
Preprocessing: Anomaly detection, semantic mask extraction
Training: SGD (lr=0.01), CrossEntropyLoss, batch_size=8
Deployment: Streamlit web app (batch upload, auto-crop, ZIP export)
```


***

## 📁 Project Structure

```
ai-vision-extract/
├── app.py                    # Streamlit web interface
├── preprocess.py            # COCO cleaning pipeline
├── train.py                 # DeepLabV3 training
├── data/
│   └── coco2017/            # Dataset (download separately)
├── model/
│   └── deeplabv3_resnet101_coco.pth  # Trained weights
├── requirements.txt
└── README.md
```


***

## 🎯 Features

✅ **Batch Upload** (JPG/PNG/JPEG)
✅ **Background Options** (Black/White/Studio Gray)
✅ **Image Enhancements** (Brightness/Contrast sliders)
✅ **Auto-Crop** to subject bounding box
✅ **Coverage Metrics** (% subject pixels)
✅ **Individual PNG + ZIP Export**
✅ **Dark Mode UI** with real-time previews

***

## 🗺️ Development Roadmap

| Week | Milestone | Deliverables |
| :-- | :-- | :-- |
| **1-2** | **Data Pipeline** | Clean masks, anomaly reports [^3] |
| **3-4** | **Model Training** | DeepLabV3 with 67.4% mIoU [^2] |
| **5-6** | **Inference** | Model loading, image processing [^4] |
| **7** | **Web UI** | Streamlit app with batch processing [^4] |
| **8** | **Documentation** | Full technical report \& demo [^1] |


***

## 📈 Results Highlights

- **Data Quality:** 99% retention after filtering 123K+ images[^3]
- **Model Performance:** Industry-competitive 67.4% mIoU on COCO segmentation[^2]
- **Production Ready:** GPU-optimized inference, batch processing[^4]
- **User Experience:** Intuitive controls, instant feedback[^4]

***

## 🔧 Troubleshooting

**❌ "Model weights not found"**

```bash
python train.py  # Train first
```

**❌ "Out of memory"**

```bash
# Reduce batch size in app.py
BATCH_SIZE = 4  # or 2
```

**❌ Slow inference**

- Use GPU: Install PyTorch CUDA
- Enable `torch.backends.cudnn.benchmark = True`

***

## 📚 Evaluation Metrics[^1]

**Primary:** Intersection over Union (IoU) - **67.4%**
**Secondary:** Dice Coefficient, Pixel Accuracy - **92.4%**
**Visual:** Before/after subject isolation quality

***

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

***

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

**⭐ Star this repo if it helps your computer vision projects!**

***

*Built with ❤️ for production ML workflows*
**Dataset:** [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)[^1]
**Model:** [DeepLabV3](https://pytorch.org/vision/stable/models.html#deeplabv3-resnet101)[^2]
**App:** [Streamlit](https://streamlit.io)[^4]
<span style="display:none">[^5]</span>

<div align="center">⁂</div>

[^1]: AI_VisionExtract.pdf

[^2]: deeplabv3_resnet101_image_seg_model_train.ipynb

[^3]: image-data-preprocessing_v3.ipynb

[^4]: app.py

[^5]: AI_VisionExtract.pdf

