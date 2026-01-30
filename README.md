AI Vision Extract: Subject Isolation via Deep Learning 🖼️✂️
Project Overview 🎯
AI Vision Extract automatically detects and extracts the main subject from any image, producing clean cutouts with customizable backgrounds. Built with DeepLabV3 ResNet101 and COCO 2017 dataset, it achieves 67.4% mIoU and 92.4% pixel accuracy.

Use Cases: Photography automation, e-commerce, AR/VR, virtual conferencing backgrounds.

🚀 Quick Start (5 Minutes)
1. Clone Repository
git clone https://github.com/yourusername/ai-vision-extract.git
cd ai-vision-extract/Project
2. Download Dataset
COCO 2017 Dataset
Extract to data/coco2017/ (train2017, val2017, annotations)
3. Setup Environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
4. Downloading Trained Model
Download this Trained model by click the link:

Model: DeepLabV3

And move into this folder:

cd ai-vision-extract/Project/model/
5. Launch Web App 🚀
streamlit run app.py
Open http://localhost:8501 → Upload images → Instant subject extraction!

📊 Key Performance Metrics
Model	Mean IoU	Pixel Accuracy
DeepLabV3-ResNet101	67.4%	92.4%
Dataset: 122,206 clean masks from COCO 2017 (99% valid after filtering)

🛠️ Technical Stack
Dataset: COCO 2017 (118K train + 5K val images)
Model: DeepLabV3 + ResNet101 backbone (81 output classes)
Preprocessing: Anomaly detection, semantic mask extraction
Training: SGD (lr=0.01), CrossEntropyLoss, batch_size=8
Deployment: Streamlit web app (batch upload, auto-crop, ZIP export)
📁 Project Structure
ai-vision-extract/Project
├── app.py                    # Streamlit web interface
├── data/
│   └── test.jpg
├── model/
│   └── deeplabv3_resnet101_coco.pth  # Trained model
├── requirements.txt

🎯 Features
✅ Batch Upload (JPG/PNG/JPEG) ✅ Background Options (Black/White/Studio Gray) ✅ Image Enhancements (Brightness/Contrast sliders) ✅ Auto-Crop to subject bounding box ✅ Coverage Metrics (% subject pixels) ✅ Individual PNG + ZIP Export ✅ Dark Mode UI with real-time previews

🗺️ Development Roadmap
Week	Milestone	Deliverables
1-2	Data Pipeline	Clean masks, anomaly reports
3-4	Model Training	DeepLabV3 with 67.4% mIoU
5-6	Inference	Model loading, image processing
7	Web UI	Streamlit app with batch processing
8	Documentation	Full technical report & demo
📈 Results Highlights
Data Quality: 99% retention after filtering 123K+ images
Model Performance: Industry-competitive 67.4% mIoU on COCO segmentation
Production Ready: GPU-optimized inference, batch processing
User Experience: Intuitive controls, instant feedback
📚 Evaluation Metrics
Primary: Intersection over Union (IoU) - 67.4% Secondary: Pixel Accuracy - 92.4% Visual: Before/after subject isolation quality

📄 License
MIT License - see LICENSE file.

⭐ Star this repo if it helps your computer vision projects!

