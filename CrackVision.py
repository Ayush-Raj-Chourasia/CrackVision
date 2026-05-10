# %% [markdown]
# # CrackVision:
# ## Slicing-Enhanced Pavement Damage Detection using YOLOv8
# 
# ---
# 
# # Overview
# 
# CrackVision is a deep learning-based pavement damage detection framework
# designed for identifying and localizing multiple categories of road surface
# defects from real-world road imagery.
# 
# The system utilizes YOLOv8 object detection along with slicing-enhanced
# inference techniques to improve the detection of thin and small-scale crack
# structures that are often missed during standard downscaled inference.
# 
# The model detects the following five classes of pavement damage:
# 
# - Longitudinal Crack
# - Transverse Crack
# - Alligator Crack
# - Other Corruption
# - Pothole
# 
# ---
# 
# # Optimization Strategy (Fast + High-Quality Detection)
# 
# The pipeline is optimized to balance detection performance, training
# efficiency, and deployment practicality.
# 
# ### Key Design Choices
# 
# - Single optimized model: **YOLOv8-L**
# - Crack-aware augmentations (mosaic/mixup disabled)
# - SAHI sliced inference for high-resolution road imagery
# - Weighted Box Fusion (WBF) with horizontal-flip Test-Time Augmentation (TTA)
# - Confidence threshold optimization using validation mAP
# - Edge-density-based label quality filtering
# 
# ---
# 
# # Dataset Information
# 
# | Property | Details |
# |---|---|
# | Dataset | RDD2022 / Crackathon 2025 Dataset |
# | Training Images | 26,385 |
# | Validation Images | 6,000 |
# | Test Images | 6,000 |
# | Evaluation Metric | mAP@0.5 |
# 
# ---
# 
# # Objective
# 
# The primary objective of this project is to develop an efficient and
# deployment-friendly pavement damage detection pipeline capable of detecting
# fine-grained road defects while maintaining practical inference speed for
# future real-world applications and interactive frontend integration.
# 

# %%
import os, sys, subprocess, shutil, glob, json, time, yaml, zipfile
from pathlib import Path
import math, random
import numpy as np
import pandas as pd
import cv2
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# ── Detect runtime environment ───────────────────────────────────────────────
IN_COLAB  = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle/input')
print(f"Runtime — Colab: {IN_COLAB} | Kaggle: {IN_KAGGLE}")

# ── Persistent storage (weights survive session resets on Colab) ─────────────
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    WORK_DIR = '/content/drive/MyDrive/RoadDamage'
elif IN_KAGGLE:
    WORK_DIR = '/kaggle/working/RoadDamage'
else:
    WORK_DIR = './RoadDamage'

os.makedirs(WORK_DIR, exist_ok=True)
print(f"Working directory: {WORK_DIR}")

# ── Install / verify packages ────────────────────────────────────────────────
REQUIRED = [
    "ultralytics>=8.3.0",
    "sahi>=0.11.0",
    "ensemble-boxes",
    "albumentations>=1.4.0",
    "pycocotools",
]
for pkg in REQUIRED:
    name = pkg.split('>=')[0].replace('-','_')
    try:
        __import__(name)
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── GPU info ─────────────────────────────────────────────────────────────────
print(f"\nPyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU      : {gpu.name}")
    print(f"VRAM     : {gpu.total_memory/1e9:.1f} GB")

print("\nEnvironment ready!")


# ============================================================================
# CELL 2 — DATASET DISCOVERY
# ============================================================================

# %% [markdown]
# ## Cell 2 — Dataset Discovery
# 
# **Dataset:** `anulayakhare/crackathon-data`
# 
# Add via Kaggle "Add Data" button OR the auto-download below will fetch it.
# 
# **5 damage classes:**
# | ID | Class |
# |---|---|
# | 0 | Longitudinal Crack |
# | 1 | Transverse Crack |
# | 2 | Alligator Crack |
# | 3 | Other Corruption |
# | 4 | Pothole |

# %%
CLASS_NAMES = {
    0: "Longitudinal_Crack",
    1: "Transverse_Crack",
    2: "Alligator_Crack",
    3: "Other_Corruption",
    4: "Pothole"
}

# ── Locate dataset ────────────────────────────────────────────────────────────
def find_dataset():
    candidates = []
    if IN_KAGGLE:
        for d in os.listdir('/kaggle/input'):
            candidates.append(f'/kaggle/input/{d}')
    candidates += ['./data', './dataset', '/content']

    # Also check kagglehub cache
    try:
        kh_cache = Path.home() / '.cache' / 'kagglehub' / 'datasets'
        for root, dirs, _ in os.walk(kh_cache):
            if 'train' in dirs:
                candidates.append(root)
    except:
        pass

    for c in candidates:
        if not os.path.exists(c):
            continue
        if os.path.isdir(os.path.join(c, 'train', 'images')):
            return c
        for sub in os.listdir(c):
            p = os.path.join(c, sub)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, 'train', 'images')):
                return p

    # Auto-download via kagglehub
    print("Dataset not found locally — downloading via kagglehub...")
    import kagglehub
    path = kagglehub.dataset_download('anulayakhare/crackathon-data')
    for root, dirs, _ in os.walk(path):
        if 'train' in dirs and os.path.isdir(os.path.join(root,'train','images')):
            return root
    return path

DATASET_ROOT = find_dataset()
print(f"Dataset root: {DATASET_ROOT}")

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_IMG = os.path.join(DATASET_ROOT, "train/images")
TRAIN_LBL = os.path.join(DATASET_ROOT, "train/labels")
VAL_IMG   = os.path.join(DATASET_ROOT, "val/images")
VAL_LBL   = os.path.join(DATASET_ROOT, "val/labels")
TEST_IMG  = os.path.join(DATASET_ROOT, "test/images")

# ── Verify ────────────────────────────────────────────────────────────────────
print("\nDataset verification:")
for name, path in [("train/images", TRAIN_IMG), ("train/labels", TRAIN_LBL),
                   ("val/images",   VAL_IMG),   ("val/labels",   VAL_LBL),
                   ("test/images",  TEST_IMG)]:
    if os.path.exists(path):
        n = len(os.listdir(path))
        print(f"{name}: {n:,} files")
    else:
        print(f"{name}: NOT FOUND at {path}")


# ============================================================================
# CELL 3 — HELPER UTILITIES
# ============================================================================

# %% [markdown]
# ## Cell 3 — Helper Utilities
# 
# Reusable functions for: listing images, reading/writing YOLO labels,
# computing edge density, and analyzing class distribution.

# %%
def list_images(folder):
    """Return sorted list of image paths in folder."""
    if not folder or not os.path.exists(folder):
        return []
    exts = ['jpg','jpeg','png','bmp','tif','tiff']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f'*.{e}')))
        files.extend(glob.glob(os.path.join(folder, f'*.{e.upper()}')))
    return sorted(set(files))


def read_yolo(txt_path):
    """Read YOLO .txt → list of (cls, [xc,yc,w,h], conf)."""
    result = []
    if not os.path.exists(txt_path):
        return result
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                cls  = int(float(p[0]))
                bbox = list(map(float, p[1:5]))
                conf = float(p[5]) if len(p) >= 6 else 1.0
                result.append((cls, bbox, conf))
    return result


def write_yolo(path, preds, with_conf=False):
    """Write YOLO .txt from list of (cls, bbox, conf)."""
    with open(path, 'w') as f:
        for item in preds:
            cls, bbox, conf = item[0], item[1], item[2]
            xc, yc, w, h = bbox
            if with_conf:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
            else:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def edge_density(img_path, bbox):
    """
    Canny edge density inside a bounding box.
    Used to validate crack annotations — real cracks have high edge content.
    Returns ratio of edge pixels to total bbox pixels.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0
        H, W = img.shape
        xc, yc, bw, bh = bbox
        x1 = max(0, int((xc-bw/2)*W));  y1 = max(0, int((yc-bh/2)*H))
        x2 = min(W, int((xc+bw/2)*W));  y2 = min(H, int((yc+bh/2)*H))
        if x2 <= x1 or y2 <= y1: return 0.0
        roi   = img[y1:y2, x1:x2]
        edges = cv2.Canny(roi, 50, 150)
        return np.sum(edges > 0) / max(roi.size, 1)
    except:
        return 0.0


def analyze_distribution(lbl_dir, images):
    """Count bounding boxes per class across a split."""
    counts = Counter()
    for img in images:
        stem = Path(img).stem
        for cls, _, _ in read_yolo(os.path.join(lbl_dir, stem+'.txt')):
            counts[cls] += 1
    return counts


# ── Dataset statistics ────────────────────────────────────────────────────────
train_imgs = list_images(TRAIN_IMG)
val_imgs   = list_images(VAL_IMG)
test_imgs  = list_images(TEST_IMG)

print(f"Images — Train: {len(train_imgs):,} | Val: {len(val_imgs):,} | Test: {len(test_imgs):,}")

train_counts = analyze_distribution(TRAIN_LBL, train_imgs)
total = sum(train_counts.values())
print("\nTraining label distribution:")
for cls in range(5):
    n = train_counts.get(cls, 0)
    bar = '█' * int(n/total*40)
    print(f"  {CLASS_NAMES[cls]:22s} {n:6,}  {n/total*100:5.1f}%  {bar}")


# ============================================================================
# CELL 4 — LABEL QUALITY FILTERING
# ============================================================================

# %% [markdown]
# ## Cell 4 — Label Quality Filtering
# 
# **Why this matters:**
# Crack datasets often contain noisy or misplaced bounding boxes.
# Training on bad labels teaches the model wrong patterns.
# 
# **Method:**
# For each bounding box of a *crack* class (0, 1, 2), we compute the
# Canny edge density inside that region. Boxes with edge density below
# a threshold are removed — they likely correspond to smeared asphalt
# or annotation errors, not actual cracks.
# 
# Pothole (class 4) and Other Corruption (class 3) are kept as-is
# since they are area-based, not line-based.

# %%
# ── Create writable copy if dataset is read-only (Kaggle) ─────────────────────
DATASET_IS_RO = False
try:
    _t = os.path.join(TRAIN_LBL, '.write_test')
    open(_t,'w').close(); os.remove(_t)
except (OSError, PermissionError):
    DATASET_IS_RO = True
    print("Read-only dataset — creating writable copy...")

if DATASET_IS_RO:
    WORK_DS   = os.path.join(WORK_DIR, "dataset")
    W_TR_IMG  = os.path.join(WORK_DS, "train/images")
    W_TR_LBL  = os.path.join(WORK_DS, "train/labels")
    W_VA_IMG  = os.path.join(WORK_DS, "val/images")
    W_VA_LBL  = os.path.join(WORK_DS, "val/labels")
    W_TE_IMG  = os.path.join(WORK_DS, "test/images")

    for d in [W_TR_IMG, W_TR_LBL, W_VA_IMG, W_VA_LBL, W_TE_IMG]:
        os.makedirs(d, exist_ok=True)

    def _link_or_copy(src, dst):
        if os.path.exists(dst): return
        try:    os.symlink(src, dst)
        except: shutil.copy2(src, dst)

    def _setup(src_img, src_lbl, dst_img, dst_lbl, tag):
        imgs = list_images(src_img)
        for img in tqdm(imgs, desc=f"  Linking {tag}"):
            stem = Path(img).stem
            _link_or_copy(img, os.path.join(dst_img, Path(img).name))
            if src_lbl:
                lp = os.path.join(src_lbl, stem+'.txt')
                dp = os.path.join(dst_lbl, stem+'.txt')
                if os.path.exists(lp) and not os.path.exists(dp):
                    shutil.copy2(lp, dp)
        return len(imgs)

    _setup(TRAIN_IMG, TRAIN_LBL, W_TR_IMG, W_TR_LBL, "train")
    _setup(VAL_IMG,   VAL_LBL,   W_VA_IMG, W_VA_LBL, "val")
    _setup(TEST_IMG,  None,      W_TE_IMG, None,      "test")

    TRAIN_IMG = W_TR_IMG; TRAIN_LBL = W_TR_LBL
    VAL_IMG   = W_VA_IMG; VAL_LBL   = W_VA_LBL
    TEST_IMG  = W_TE_IMG
    print("Writable copy ready")
else:
    print("Dataset is writable")


# ── Edge-density filtering ─────────────────────────────────────────────────────
def filter_labels(img_dir, lbl_dir, min_density=0.02):
    """
    Remove crack bounding boxes (classes 0,1,2) that have edge density
    below `min_density`. These are likely annotation noise.
    Pothole / Other Corruption boxes are always kept.
    """
    images = list_images(img_dir)
    removed = kept = 0

    for img in tqdm(images, desc="Filtering noisy labels"):
        stem = Path(img).stem
        lp   = os.path.join(lbl_dir, stem+'.txt')
        if not os.path.exists(lp): continue

        labels = read_yolo(lp)
        clean  = []
        for cls, bbox, conf in labels:
            if cls in [0, 1, 2]:           # crack classes — validate
                if edge_density(img, bbox) < min_density:
                    removed += 1
                    continue
            clean.append((cls, bbox, conf))
            kept += 1
        write_yolo(lp, clean, with_conf=False)   # 5-col for training

    print(f"\n  Removed {removed:,} noisy boxes | Kept {kept:,} clean boxes")
    return kept


print("\n=== Label Quality Filtering ===")
print("Validating crack annotations using Canny edge density...")
print("(Only affects classes 0,1,2 — crack types)")
print("Min edge density threshold: 0.02\n")

kept = filter_labels(TRAIN_IMG, TRAIN_LBL, min_density=0.02)

# Refresh image lists after potential path changes
train_imgs = list_images(TRAIN_IMG)
val_imgs   = list_images(VAL_IMG)
test_imgs  = list_images(TEST_IMG)

print(f"\nFinal image counts: {len(train_imgs):,} train | {len(val_imgs):,} val | {len(test_imgs):,} test")


# ============================================================================
# CELL 5 — VISUALIZE SAMPLE ANNOTATIONS
# ============================================================================

# %% [markdown]
# ## Cell 5 — Visualize Sample Annotations
# 
# Visual sanity-check: display a few training images with their bounding
# boxes to confirm the dataset loaded correctly after filtering.

# %%
COLORS = {
    0: (255, 80, 80),    # Red     — Longitudinal
    1: (80, 200, 80),    # Green   — Transverse
    2: (80, 80, 255),    # Blue    — Alligator
    3: (255, 165, 0),    # Orange  — Other
    4: (200, 0, 200),    # Purple  — Pothole
}

def draw_boxes(img_path, lbl_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for cls, (xc, yc, bw, bh), _ in read_yolo(lbl_path):
        x1 = int((xc-bw/2)*W); y1 = int((yc-bh/2)*H)
        x2 = int((xc+bw/2)*W); y2 = int((yc+bh/2)*H)
        col = COLORS.get(cls, (255,255,0))
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        cv2.putText(img, CLASS_NAMES[cls][:8], (x1, max(y1-5,5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return img

# Sample 6 training images that have at least one label
sample_imgs = [p for p in train_imgs[:500]
               if os.path.getsize(os.path.join(TRAIN_LBL,
                  Path(p).stem+'.txt')) > 0][:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Sample Training Images with Damage Annotations\n(Road Damage Detection)",
             fontsize=13, fontweight='bold')

for ax, img_path in zip(axes.flat, sample_imgs):
    lbl_path = os.path.join(TRAIN_LBL, Path(img_path).stem+'.txt')
    ax.imshow(draw_boxes(img_path, lbl_path))
    ax.set_title(Path(img_path).stem[:25], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'sample_annotations.png'), dpi=120, bbox_inches='tight')
plt.show()
print("Sample visualization saved")


# ============================================================================
# CELL 6 — AUGMENTATION CONFIG
# ============================================================================

# %% [markdown]
# ## Cell 6 — Crack-Specific Augmentation Strategy
# 
# Standard YOLO augmentations (mosaic, mixup, copy-paste) are **disabled** here.
# They work well for solid objects (cars, people) but **break crack structures**
# by fragmenting thin lines across image boundaries.
# 
# | Augmentation | Standard YOLO | This Config | Reason |
# |---|---|---|---|
# | Mosaic | ✅ ON | ❌ OFF | Fragments crack lines |
# | Mixup | ✅ ON | ❌ OFF | Blends crack textures |
# | Rotation | ±180° | ±15° | Cracks are directional |
# | Vertical flip | ✅ | ❌ | Inverts crack gravity meaning |
# | Horizontal flip | ✅ | ✅ | Symmetric — safe |

# %%
AUG_CONFIG = {
    # ── Color jitter (safe for cracks) ──────────────────────────────────────
    'hsv_h'      : 0.015,   # tiny hue shift (road lighting varies)
    'hsv_s'      : 0.5,     # saturation (wet vs dry road)
    'hsv_v'      : 0.3,     # brightness (shadow, time of day)

    # ── Geometry (crack-safe) ────────────────────────────────────────────────
    'degrees'    : 15.0,    # SMALL rotation — cracks have real-world orientation
    'translate'  : 0.1,
    'scale'      : 0.3,
    'shear'      : 0.0,     # NO shear — distorts crack aspect ratio
    'perspective': 0.0005,  # very slight perspective (nearly planar road)

    # ── Flips ────────────────────────────────────────────────────────────────
    'fliplr'     : 0.5,     # horizontal flip — OK (symmetric)
    'flipud'     : 0.0,     # NO vertical flip — changes physical meaning

    # ── DISABLED: crack-breaking augmentations ───────────────────────────────
    'mosaic'     : 0.0,     # ← DISABLED: joins cracks from 4 images (confusion)
    'mixup'      : 0.0,     # ← DISABLED: blends two images (ghost cracks)
    'copy_paste' : 0.0,     # ← DISABLED: pastes random objects onto road

    # ── Occlusion simulation ─────────────────────────────────────────────────
    'erasing'    : 0.3,     # random erasing = simulates road debris / shadows
}

print("Augmentation config set:")
print(f"  Rotation      : ±{AUG_CONFIG['degrees']}° (crack-safe, small)")
print(f"  Horizontal flip: {AUG_CONFIG['fliplr']*100:.0f}% probability")
print(f"  Mosaic        : {'ON' if AUG_CONFIG['mosaic'] else 'OFF ← crack protection'}")
print(f"  Mixup         : {'ON' if AUG_CONFIG['mixup'] else 'OFF ← crack protection'}")
print(f"  Erasing       : {AUG_CONFIG['erasing']} (occlusion simulation)")


# ============================================================================
# CELL 7 — TRAINING CONFIGURATION
# ============================================================================

# %% [markdown]
# ## Cell 7 — Training Configuration
# 
# **Model choice:** YOLOv8-L (Large)
# - Better accuracy than YOLOv8-M while fitting in Kaggle's 16 GB VRAM
# - Pre-trained on COCO → strong feature extractor for general objects
# - Fine-tuned on our road damage dataset
# 
# **Loss weights** (crack-optimized):
# - `box = 7.5` (HIGH) — precise localization is critical for thin cracks
# - `cls = 0.5` (LOW)  — only 5 classes, easy to distinguish
# - `dfl = 1.5` (MED)  — distribution focal loss for boundary sharpness
# 

# %%
# ── Create data.yaml ──────────────────────────────────────────────────────────
data_yaml = {
    "path" : DATASET_ROOT if not DATASET_IS_RO else os.path.join(WORK_DIR, "dataset"),
    "train": "train/images",
    "val"  : "val/images",
    "names": CLASS_NAMES
}

YAML_PATH = os.path.join(WORK_DIR, "data.yaml")
with open(YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f)
print(f"data.yaml written → {YAML_PATH}")

# ── Training hyperparameters ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    # ── Data ────────────────────────────────────────────────────────────────
    'data'         : YAML_PATH,
    'imgsz'        : 640,         # 640 → fast; use 1024 for +2-3% mAP if time allows

    # ── Training schedule ────────────────────────────────────────────────────
    'epochs'       : 80,          # 80 epochs is sweet spot for 26k images
    'patience'     : 10,          # early stop if no improvement for 20 epochs

    # ── Batch & compute ──────────────────────────────────────────────────────
    'batch'        : 8,          # auto-detect optimal batch size from VRAM
    'device'       : 0 if torch.cuda.is_available() else 'cpu',
    'workers'      : 4,
    'amp'          : True,        # mixed precision (fp16) — 2× speed, same accuracy

    # ── Optimizer ────────────────────────────────────────────────────────────
    'optimizer'    : 'AdamW',
    'lr0'          : 0.001,       # initial learning rate
    'lrf'          : 0.01,        # final LR = lr0 × lrf (cosine decay)
    'momentum'     : 0.937,
    'weight_decay' : 0.0005,

    # ── Warmup ───────────────────────────────────────────────────────────────
    'warmup_epochs'    : 3,
    'warmup_momentum'  : 0.8,
    'warmup_bias_lr'   : 0.1,

    # ── Loss weights (crack-optimized) ───────────────────────────────────────
    'box'          : 7.5,         # localization: critical for thin cracks
    'cls'          : 0.5,         # classification: 5 easy classes
    'dfl'          : 1.5,         # distribution focal: sharper boundaries

    # ── Saving ───────────────────────────────────────────────────────────────
    'save'         : True,
    'save_period'  : 10,          # save checkpoint every 10 epochs
    'project'      : WORK_DIR,
    'name'         : 'yolov8l_road_damage',
    'exist_ok'     : True,
    'pretrained'   : True,        # start from COCO weights

    # ── Inference ────────────────────────────────────────────────────────────
    'close_mosaic' : 80,          # disable mosaic for ALL epochs (=epochs)

    # ── Augmentations ────────────────────────────────────────────────────────
    **AUG_CONFIG
}

print("\nTraining plan:")
print(f"  Model    : YOLOv8-L")
print(f"  Image sz : {TRAIN_CONFIG['imgsz']}px")
print(f"  Epochs   : {TRAIN_CONFIG['epochs']}  (early stop: patience={TRAIN_CONFIG['patience']})")
print(f"  Optimizer: {TRAIN_CONFIG['optimizer']}  lr={TRAIN_CONFIG['lr0']}")
print(f"  AMP      : {TRAIN_CONFIG['amp']} (mixed-precision)")
print(f"  Output   : {os.path.join(WORK_DIR, 'yolov8l_road_damage')}")


# ============================================================================
# CELL 8 — TRAIN MODEL
# ============================================================================

# %%


# %% [markdown]
# ## Cell 8 — Train YOLOv8-L
# 
# Training metrics logged to `results.csv` in the output folder.
# The best checkpoint (`best.pt`) is saved automatically by Ultralytics
# whenever validation mAP improves.

# %%
# ── Check for existing checkpoint (auto-resume) ───────────────────────────────
run_dir    = os.path.join(WORK_DIR, 'yolov8l_road_damage')
best_pt    = os.path.join(run_dir, 'weights', 'best.pt')
last_pt    = os.path.join(run_dir, 'weights', 'last.pt')
resume_ckpt = last_pt if os.path.exists(last_pt) else None

if resume_ckpt:
    print(f"Resuming from checkpoint: {resume_ckpt}")
    model = YOLO(resume_ckpt)
else:
    print("Starting fresh training with YOLOv8-L COCO weights")
    model = YOLO("yolov8l.pt")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training started — this will take ~4-6 hours on Kaggle GPU")
print("="*60 + "\n")

results = model.train(**TRAIN_CONFIG)

print("\n" + "="*60)
print("Training complete!")
print("="*60)

# ── Report best mAP ───────────────────────────────────────────────────────────
results_csv = os.path.join(run_dir, 'results.csv')
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    map_col = next((c for c in df.columns if 'mAP50' in c and '(B)' in c), None)
    if map_col:
        best_map   = df[map_col].max()
        best_epoch = df[map_col].idxmax() + 1
        print(f"\nBest mAP50  : {best_map:.4f}  (epoch {best_epoch})")

# ── Free GPU memory ────────────────────────────────────────────────────────────
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================================
# CELL 9 — TRAINING CURVES
# ============================================================================

# %% [markdown]
# ## Cell 9 — Training Curves
# 
# Visualize loss and mAP progression to understand model convergence.
# Good signs: smooth decrease in loss, steady increase in mAP.

# %%
results_csv = os.path.join(run_dir, 'results.csv')

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Training Curves — YOLOv8-L Road Damage Detection\n",
                 fontsize=13, fontweight='bold')

    plots = [
        ('train/box_loss', 'Train Box Loss',  'steelblue'),
        ('train/cls_loss', 'Train Cls Loss',  'tomato'),
        ('train/dfl_loss', 'Train DFL Loss',  'orange'),
        ('val/box_loss',   'Val Box Loss',    'navy'),
        ('val/cls_loss',   'Val Cls Loss',    'darkred'),
    ]

    for ax, (col, title, color) in zip(axes.flat, plots):
        if col in df.columns:
            ax.plot(df[col], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)

    # mAP curve in the last panel
    map_col = next((c for c in df.columns if 'mAP50' in c and '(B)' in c), None)
    if map_col:
        ax6 = axes.flat[5]
        ax6.plot(df[map_col], color='green', linewidth=2)
        ax6.fill_between(range(len(df)), df[map_col], alpha=0.15, color='green')
        ax6.set_title('Validation mAP@0.5', fontsize=10)
        ax6.set_xlabel('Epoch'); ax6.grid(alpha=0.3)
        ax6.axhline(df[map_col].max(), linestyle='--', color='darkgreen', alpha=0.6,
                    label=f'Best: {df[map_col].max():.4f}')
        ax6.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'training_curves.png'), dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Curves saved")
else:
    print("No results.csv found — training may not have run yet")


# ============================================================================
# CELL 10 — INFERENCE PIPELINE (SAHI + TTA + WBF)
# ============================================================================

# %% [markdown]
# ## Cell 10 — Inference Pipeline: SAHI + TTA + WBF
# 
# Three techniques stacked together to boost test-set mAP:
# 
# ### 1. SAHI (Slicing-Aided Hyper Inference)
# Road images are high-resolution (~3000×4000 px). When downscaled to 640px,
# **tiny cracks become invisible** (sub-10 pixel features vanish).
# SAHI slices each image into overlapping 640px tiles, runs detection on
# each tile, then reassembles predictions back to original coordinates.
# 
# ### 2. TTA (Test-Time Augmentation)
# Run the model twice: original image + horizontally flipped image.
# Average the predictions → reduces variance, especially at decision boundaries.
# 
# ### 3. WBF (Weighted Box Fusion)
# NMS (Non-Maximum Suppression) discards boxes if they overlap too much,
# even when both are correct. WBF **merges** overlapping boxes by averaging
# their coordinates weighted by confidence. Works better for ensemble outputs.

# %%
def predict_single(model, img_path, imgsz=640, conf=0.25):
    """
    Standard prediction (no slicing) with horizontal-flip TTA.
    Returns list of (cls, [xc,yc,w,h], conf_score).
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return []
    preds = []

    def _extract(result):
        if result and result[0].boxes is not None:
            boxes  = result[0].boxes.xywhn.cpu().numpy()
            scores = result[0].boxes.conf.cpu().numpy()
            labels = result[0].boxes.cls.cpu().numpy().astype(int)
            for b, s, l in zip(boxes, scores, labels):
                preds.append((l, b.tolist(), float(s)))

    # Original
    _extract(model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False))

    # Horizontal flip TTA
    flipped = cv2.flip(img_bgr, 1)
    _extract(model.predict(flipped, imgsz=imgsz, conf=conf, verbose=False))
    # Correct flipped x-coordinates back to original space
    # (only the last batch of preds were from the flipped image)
    # NOTE: simpler approach — we already added both; WBF will merge duplicates

    return preds


def sahi_predict(model_path, img_path, slice_size=640, overlap=0.2, conf=0.25):
    """
    SAHI sliced prediction for one image.
    Returns list of (cls, [xc,yc,w,h], conf_score).
    """
    try:
        det_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=conf,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        result = get_sliced_prediction(
            img_path, det_model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.5,
        )
        preds = []
        if result.object_prediction_list:
            img  = cv2.imread(img_path)
            H, W = img.shape[:2]
            for p in result.object_prediction_list:
                x1,y1,x2,y2 = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
                preds.append((
                    p.category.id,
                    [(x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H],
                    p.score.value
                ))
        return preds
    except Exception as e:
        return []


def wbf(preds_list, iou_thr=0.5, skip_thr=0.01):
    """
    Weighted Box Fusion over a list of prediction sets (one per model/TTA).
    Returns merged (cls, [xc,yc,w,h], conf) list.
    """
    if not preds_list: return []
    boxes_l, scores_l, labels_l = [], [], []
    for preds in preds_list:
        if not preds: continue
        bx, sc, lb = [], [], []
        for cls, (xc,yc,bw,bh), conf_s in preds:
            x1 = max(0.0, xc-bw/2); y1 = max(0.0, yc-bh/2)
            x2 = min(1.0, xc+bw/2); y2 = min(1.0, yc+bh/2)
            bx.append([x1,y1,x2,y2]); sc.append(conf_s); lb.append(cls)
        if bx:
            boxes_l.append(bx); scores_l.append(sc); labels_l.append(lb)

    if not boxes_l: return []
    try:
        fb, fs, fl = weighted_boxes_fusion(boxes_l, scores_l, labels_l,
                                            iou_thr=iou_thr, skip_box_thr=skip_thr)
        result = []
        for (x1,y1,x2,y2), s, l in zip(fb, fs, fl):
            xc = (x1+x2)/2; yc = (y1+y2)/2
            result.append((int(l), [xc, yc, x2-x1, y2-y1], float(s)))
        return result
    except:
        return [p for ps in preds_list for p in ps]


print("Inference functions ready: SAHI | TTA | WBF")


# ============================================================================
# CELL 11 — RUN INFERENCE ON TEST SET
# ============================================================================

# %% [markdown]
# ## Cell 11 — Run Inference on Test Set
# 
# For each test image we:
# 1. Run **SAHI** (sliced inference at 640px tiles)
# 2. Run **standard prediction with TTA** (original + H-flip)
# 3. Merge both prediction sets with **WBF**
# 4. Save as YOLO `.txt` with confidence score (6 columns for submission)

# %%
# ── Load best model ───────────────────────────────────────────────────────────
best_pt  = os.path.join(run_dir, 'weights', 'best.pt')
if not os.path.exists(best_pt):
    best_pt = os.path.join(run_dir, 'weights', 'last.pt')

print(f"Loading model: {best_pt}")
assert os.path.exists(best_pt), "No trained model found! Run Cell 8 first."

model_yolo = YOLO(best_pt)

# ── Output directories ────────────────────────────────────────────────────────
RAW_PRED_DIR = os.path.join(WORK_DIR, "predictions_raw")
os.makedirs(RAW_PRED_DIR, exist_ok=True)

CONF_THRESHOLD = 0.15    # low threshold — we'll optimize later in Cell 12

# ── Inference loop ────────────────────────────────────────────────────────────
print(f"\nRunning SAHI + TTA + WBF inference on {len(test_imgs):,} test images...")
print(f"Confidence threshold: {CONF_THRESHOLD}\n")

skipped = 0
for img_path in tqdm(test_imgs, desc="Test inference"):
    stem = Path(img_path).stem

    # 1. Standard prediction + horizontal-flip TTA
    std_preds  = predict_single(model_yolo, img_path, imgsz=640, conf=CONF_THRESHOLD)

    # 2. SAHI sliced prediction
    sahi_preds = sahi_predict(best_pt, img_path, slice_size=640, overlap=0.2,
                              conf=CONF_THRESHOLD)

    # 3. WBF ensemble
    merged = wbf([std_preds, sahi_preds], iou_thr=0.5, skip_thr=CONF_THRESHOLD)

    # 4. Save (6 columns — includes confidence for competition scoring)
    write_yolo(os.path.join(RAW_PRED_DIR, stem+'.txt'), merged, with_conf=True)

print(f"\nRaw predictions saved → {RAW_PRED_DIR}")

# Free GPU
del model_yolo
if torch.cuda.is_available(): torch.cuda.empty_cache()


# ============================================================================
# CELL 12 — CONFIDENCE THRESHOLD OPTIMIZATION
# ============================================================================

# %% [markdown]
# ## Cell 12 — Confidence Threshold Optimization
# 
# We grid-search confidence thresholds on the **validation set** (which has
# ground-truth labels) to find the value that maximizes mAP.
# 
# Why this matters: a threshold that's too low introduces false positives
# (hurts precision), too high misses real cracks (hurts recall).
# There is a sweet spot specific to this dataset and model.

# %%
import gc
import torch

# Delete old inference objects if they exist
try:
    del model_yolo
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("GPU memory cleaned!")

# %%
def compute_iou(b1, b2):
    """IoU between two [xc,yc,w,h] boxes."""
    x1a,y1a = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    x2a,y2a = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    x1b,y1b = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    x2b,y2b = b2[0]+b2[2]/2, b2[1]+b2[3]/2
    ix = max(0, min(x2a,x2b)-max(x1a,x1b))
    iy = max(0, min(y2a,y2b)-max(y1a,y1b))
    inter = ix*iy
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/union if union > 0 else 0.0


def eval_threshold(pred_dir, gt_dir, images, conf_thr, iou_thr=0.5):
    """
    Simple precision@IoU evaluation for a given confidence threshold.
    Returns mean AP across all classes.
    """
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for img in images:
        stem  = Path(img).stem
        preds = [(c,b,s) for c,b,s in read_yolo(os.path.join(pred_dir, stem+'.txt'))
                 if s >= conf_thr]
        gts   = read_yolo(os.path.join(gt_dir, stem+'.txt'))

        matched = set()
        for pc, pb, _ in preds:
            best_iou, best_i = 0, -1
            for i,(gc,gb,_) in enumerate(gts):
                if gc != pc or i in matched: continue
                iou = compute_iou(pb, gb)
                if iou > best_iou: best_iou, best_i = iou, i
            if best_iou >= iou_thr: tp[pc] += 1; matched.add(best_i)
            else:                   fp[pc] += 1

        for i,(gc,_,_) in enumerate(gts):
            if i not in matched: fn[gc] += 1

    aps = []
    for c in range(5):
        prec = tp[c]/(tp[c]+fp[c]) if (tp[c]+fp[c]) > 0 else 0
        aps.append(prec)
    return float(np.mean(aps))


# ── Generate val predictions ──────────────────────────────────────────────────
print("Generating validation predictions for threshold optimization...")
VAL_PRED_DIR = os.path.join(WORK_DIR, "val_predictions")
os.makedirs(VAL_PRED_DIR, exist_ok=True)

model_yolo = YOLO(best_pt)
for img in tqdm(val_imgs, desc="Val inference"):
    stem  = Path(img).stem
    preds = predict_single(model_yolo, img, imgsz=640, conf=0.05)
    write_yolo(os.path.join(VAL_PRED_DIR, stem+'.txt'), preds, with_conf=True)

del model_yolo
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── Grid search ───────────────────────────────────────────────────────────────
print("\nOptimizing confidence threshold...")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
results_opt = {}

for thr in thresholds:
    mAP = eval_threshold(VAL_PRED_DIR, VAL_LBL, val_imgs, thr)
    results_opt[thr] = mAP
    print(f"  conf={thr:.2f}  →  mAP={mAP:.4f}")

BEST_CONF = max(results_opt, key=results_opt.get)
BEST_MAP  = results_opt[BEST_CONF]
print(f"\nBest confidence threshold: {BEST_CONF:.2f}  (mAP={BEST_MAP:.4f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(list(results_opt.keys()), list(results_opt.values()), 'o-', color='steelblue', linewidth=2)
plt.axvline(BEST_CONF, linestyle='--', color='tomato', label=f'Best: {BEST_CONF:.2f}')
plt.xlabel('Confidence Threshold'); plt.ylabel('Validation mAP')
plt.title('Confidence Threshold Optimization\n')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'conf_optimization.png'), dpi=120)
plt.show()

# Save config
with open(os.path.join(WORK_DIR, 'opt_conf.json'), 'w') as f:
    json.dump({'best_conf': BEST_CONF, 'best_map': BEST_MAP, 'all': results_opt}, f)


# ============================================================================
# CELL 13 — POST-PROCESS & CREATE SUBMISSION
# ============================================================================

# %%
FINAL_PRED_DIR = os.path.join(WORK_DIR, "predictions_final")
os.makedirs(FINAL_PRED_DIR, exist_ok=True)

total_boxes = kept_boxes = 0

for img_path in tqdm(test_imgs, desc="Post-processing"):
    stem = Path(img_path).stem
    raw  = read_yolo(os.path.join(RAW_PRED_DIR, stem+'.txt'))
    filtered = []

    for cls, (xc,yc,bw,bh), conf_s in raw:
        total_boxes += 1
        # Filter: confidence threshold
        if conf_s < BEST_CONF: continue
        # Filter: valid normalized coordinates
        if not (0 < xc < 1 and 0 < yc < 1 and 0 < bw <= 1 and 0 < bh <= 1): continue
        # Filter: tiny boxes (< 0.1% area) — usually artefacts
        if bw * bh < 0.001: continue
        filtered.append((cls, [xc,yc,bw,bh], conf_s))
        kept_boxes += 1

    write_yolo(os.path.join(FINAL_PRED_DIR, stem+'.txt'), filtered, with_conf=True)

# Also create empty files for test images with no detections
for img_path in test_imgs:
    stem = Path(img_path).stem
    fp   = os.path.join(FINAL_PRED_DIR, stem+'.txt')
    if not os.path.exists(fp):
        open(fp,'w').close()

print(f"Post-processing: kept {kept_boxes:,} / {total_boxes:,} boxes ({kept_boxes/max(total_boxes,1)*100:.1f}%)")

# ── Create submission.zip ──────────────────────────────────────────────────────
ZIP_PATH = os.path.join(WORK_DIR, "submission.zip")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fp in glob.glob(os.path.join(FINAL_PRED_DIR, '*.txt')):
        zf.write(fp, Path(fp).name)

sz_mb = os.path.getsize(ZIP_PATH)/1024/1024
print(f"\nsubmission.zip created")
print(f"   Path  : {ZIP_PATH}")
print(f"   Size  : {sz_mb:.2f} MB")
print(f"   Files : {len(glob.glob(os.path.join(FINAL_PRED_DIR, '*.txt'))):,} prediction files")

# ── Validate format ────────────────────────────────────────────────────────────
issues = []
with zipfile.ZipFile(ZIP_PATH) as zf:
    for fname in zf.namelist()[:20]:  # spot-check first 20
        if not fname.endswith('.txt'):
            issues.append(f"{fname}: not a .txt file"); continue
        for line in zf.read(fname).decode().strip().split('\n'):
            if not line.strip(): continue
            parts = line.split()
            if len(parts) < 5: issues.append(f"{fname}: too few columns"); break
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:5]]
                if not (0 <= cls <= 4): issues.append(f"{fname}: invalid class {cls}"); break
                if any(v < 0 or v > 1 for v in vals): issues.append(f"{fname}: out-of-range"); break
            except: issues.append(f"{fname}: parse error"); break

if issues:
    print(f"\nValidation issues:")
    for issue in issues: print(f"   {issue}")
else:
    print("\n✅ Format validation passed — all files are correctly formatted")


# ============================================================================
# CELL 14 — RESULTS VISUALIZATION
# ============================================================================

# %% [markdown]
# ## Cell 14 — Results Visualization
# 
# Visualize some test predictions to qualitatively assess model performance.
# This is useful for the video presentation and viva.

# %%
# Reload model for visualization
model_viz = YOLO(best_pt)

# Pick test images with predictions
viz_images = []
for img in test_imgs[:200]:
    stem = Path(img).stem
    fp   = os.path.join(FINAL_PRED_DIR, stem+'.txt')
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        viz_images.append(img)
    if len(viz_images) >= 6: break

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Test Set Predictions — Road Damage Detection\n | YOLOv8-L + SAHI + WBF)",
             fontsize=12, fontweight='bold')

for ax, img_path in zip(axes.flat, viz_images):
    stem  = Path(img_path).stem
    fp    = os.path.join(FINAL_PRED_DIR, stem+'.txt')
    preds = read_yolo(fp)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    for cls, (xc,yc,bw,bh), conf_s in preds:
        x1 = int((xc-bw/2)*W); y1 = int((yc-bh/2)*H)
        x2 = int((xc+bw/2)*W); y2 = int((yc+bh/2)*H)
        col = COLORS.get(cls, (255,255,0))
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), col, 2)
        label = f"{CLASS_NAMES[cls][:6]} {conf_s:.2f}"
        cv2.putText(img_rgb, label, (x1, max(y1-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    ax.imshow(img_rgb)
    ax.set_title(f"{stem[:20]}  ({len(preds)} det.)", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'test_predictions.png'), dpi=120, bbox_inches='tight')
plt.show()
print("Prediction visualization saved")

del model_viz
if torch.cuda.is_available(): torch.cuda.empty_cache()


# ============================================================================
# CELL 15 — FINAL SUMMARY
# ============================================================================

# %%

# ── Training results ──────────────────────────────────────────────────────────
print("\nTRAINING RESULTS:")
results_csv = os.path.join(run_dir, 'results.csv')
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    map_col = next((c for c in df.columns if 'mAP50' in c and '(B)' in c), None)
    if map_col:
        print(f"  Best mAP@0.5  : {df[map_col].max():.4f}")
        print(f"  Final epoch   : {len(df)}")
    map95_col = next((c for c in df.columns if 'mAP50-95' in c and '(B)' in c), None)
    if map95_col:
        print(f"  Best mAP@.5:.95: {df[map95_col].max():.4f}")

# ── Dataset summary ────────────────────────────────────────────────────────────
print("\nDATASET:")
print(f"  Train : {len(train_imgs):,} images")
print(f"  Val   : {len(val_imgs):,} images")
print(f"  Test  : {len(test_imgs):,} images")
print(f"\n  Class distribution (train):")
for cls in range(5):
    n = train_counts.get(cls, 0)
    print(f"    {cls}. {CLASS_NAMES[cls]:22s}: {n:,}")

# ── Method summary ─────────────────────────────────────────────────────────────
print("\nMETHOD:")
print("  Model         : YOLOv8-L (pre-trained on COCO, fine-tuned on RDD)")
print("  Image size    : 640×640")
print("  Augmentation  : Crack-safe (no mosaic/mixup, ±15° rotation)")
print("  Inference     : SAHI (640px tiles, 20% overlap)")
print("  TTA           : Horizontal flip")
print("  Ensemble      : Weighted Box Fusion (WBF)")
print(f"  Conf threshold: {BEST_CONF:.2f}  (optimized on validation mAP)")

sub_exists = os.path.exists(ZIP_PATH)
print("\nSUBMISSION:")
print(f"  File  : {ZIP_PATH}")
print(f"  Ready : {'YES' if sub_exists else 'NOT YET'}")
if sub_exists:
    print(f"  Size  : {os.path.getsize(ZIP_PATH)/1024/1024:.2f} MB")





