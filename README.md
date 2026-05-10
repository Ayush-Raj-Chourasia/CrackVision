---
title: CrackVision
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: docker
app_file: app.py
---

# CrackVision
### Slicing-Enhanced Pavement Damage Detection using YOLOv8

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/AlphaCalculus/crackvision)

---

## 🎯 Overview

CrackVision is a deep learning pipeline for automated road damage detection and classification. It detects five types of pavement damage from real-world road imagery using **YOLOv8-L** combined with **SAHI sliced inference**, **Test-Time Augmentation (TTA)**, and **Weighted Box Fusion (WBF)**.

The key insight: standard object detectors downscale high-resolution road images to 640×640px, causing thin cracks (sub-10px features) to become invisible. CrackVision solves this by slicing images into overlapping 640px tiles, detecting on each tile, then reassembling detections.

### Live Demo
**[Hugging Face Spaces](https://huggingface.co/spaces/AlphaCalculus/crackvision)**  
**[Web Frontend on Vercel](https://crackvision.vercel.app)**

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation mAP@0.5 | **0.5082** |
| Validation mAP@0.5:0.95 | **0.2390** |
| Training epochs | 20           |
| Optimized confidence threshold | 0.50 |
| Inference speed (T4 GPU) | ~400ms/image (with SAHI) |

### Training Curves
All losses decreased smoothly over 20 epochs with mAP still rising at termination — indicating the model would improve further with additional training time.

---

## 🗂️ Dataset

**Crackathon 2025** (`anulayakhare/crackathon-data`) — Road damage images

| Split | Images |
|---|---|
| Train | 26,385 |
| Validation | 6,000 |
| Test | 6,000 |

### Damage Classes

| ID | Class | Distribution | Severity |
|---|---|---|---|
| 0 | Longitudinal Crack | 39.7% | Moderate |
| 1 | Transverse Crack | 18.1% | Moderate |
| 2 | Alligator Crack | 16.1% | High |
| 3 | Other Corruption | 16.2% | Moderate |
| 4 | Pothole | 9.9% | High |

---

## 🏗️ Architecture & Method

```
Road Image (high-res)
       │
       ├─► SAHI Slicing (640px tiles, 20% overlap)
       │         │
       │    YOLOv8-L inference per tile
       │         │
       │    Reassemble to original coords
       │
       ├─► Standard inference + H-flip TTA
       │
       └─► Weighted Box Fusion (WBF) → Final detections
```

### Key Design Decisions

**1. Crack-safe augmentations**
Standard YOLO augmentations (mosaic, mixup, copy-paste) fragment thin crack lines across image boundaries. We disable them entirely.

| Augmentation | Standard | CrackVision | Why |
|---|---|---|---|
| Mosaic | ON | OFF | Breaks crack continuity |
| Mixup | ON | OFF | Creates ghost cracks |
| Rotation | ±180° | ±15° | Cracks are directional |
| Vertical flip | ON | OFF | Changes physical meaning |

**2. Label quality filtering**
Crack bounding boxes with Canny edge density below 0.02 are removed — they represent annotation noise rather than actual cracks. 37,900 high-quality boxes retained.

**3. SAHI sliced inference**
High-resolution road images compressed to 640px lose thin crack features. SAHI tiles each image into overlapping 640px patches, allowing the detector to see cracks at their native scale.

**4. Weighted Box Fusion over NMS**
NMS discards overlapping predictions from SAHI tiles and TTA. WBF merges them by averaging coordinates weighted by confidence score.

---

## 📁 Repository Structure

```
CrackVision/
├── notebook/
│   └── crackvision_training.py     # Full training pipeline (Kaggle)
├── app/
│   ├── app.py                      # Gradio app (Hugging Face Spaces)
│   └── requirements.txt
├── frontend/
│   └── index.html                  # Web frontend (Vercel)
├── report/
│   └── report.tex                  # IEEE LaTeX report
├── assets/
│   ├── training_curves.png
│   ├── conf_optimization.png
│   └── test_predictions.png
└── README.md
```

---

## 🚀 Deployment

### Option 1: Hugging Face Spaces (Recommended)

```bash
# 1. Create new Space at huggingface.co/new-space
# 2. Select SDK: Gradio
# 3. Upload these files:
#    - app/app.py
#    - app/requirements.txt
#    - best.pt  (your trained model)
# 4. Space auto-builds → public URL
```

### Option 2: Vercel Frontend

```bash
# 1. Fork this repo
# 2. Connect to Vercel at vercel.com/new
# 3. Set environment variable:
#    HF_SPACE_URL = https://your-username-crackvision.hf.space
# 4. Deploy — Vercel builds automatically
```

### Option 3: Run Locally

```bash
# Clone
git clone https://github.com/Ayush-Raj-Chourasia/CrackVision
cd CrackVision

# Install
pip install ultralytics gradio opencv-python sahi ensemble-boxes

# Place your best.pt in the root directory

# Run Gradio app
python app/app.py

# OR run Streamlit app
pip install streamlit
streamlit run app/streamlit_app.py
```

---

## 🔬 Technical Details

### Model
- **Architecture**: YOLOv8-L (Large) — 43.7M parameters
- **Pre-training**: COCO (80 classes, 118k images)
- **Fine-tuning**: Crackathon 2025 dataset

### Training Configuration
```python
epochs       = 80   (stopped at 20 due to time constraints)
batch        = 8
imgsz        = 640
optimizer    = AdamW
lr0          = 0.001
lrf          = 0.01   # cosine decay
weight_decay = 0.0005
amp          = True   # mixed precision fp16
loss_box     = 7.5    # high — precise crack localization
loss_cls     = 0.5    # low — only 5 classes
loss_dfl     = 1.5    # medium
```


---

## 📚 References

1. Wang, C. Y., et al. "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information." arXiv, 2024.
2. Jocher, G., et al. "Ultralytics YOLOv8." GitHub, 2023. https://github.com/ultralytics/ultralytics
3. Akyon, F. C., et al. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." ICIP, 2022.
4. Solovyev, R., et al. "Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models." Image and Vision Computing, 2021.
5. Arya, D., et al. "RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection." arXiv, 2022.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

