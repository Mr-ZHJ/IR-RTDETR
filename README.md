# IR-RTDETR: Infrared Small UAV Detection via Improved RT-DETR

<p align="center">
  <a href="https://github.com/Mr-ZHJ/IR-RTDETR/stargazers"><img src="https://img.shields.io/github/stars/Mr-ZHJ/IR-RTDETR?style=flat-square&color=FFD700" alt="Stars"></a>
  <a href="https://github.com/Mr-ZHJ/IR-RTDETR/network/members"><img src="https://img.shields.io/github/forks/Mr-ZHJ/IR-RTDETR?style=flat-square&color=blue" alt="Forks"></a>
  <a href="https://github.com/Mr-ZHJ/IR-RTDETR/issues"><img src="https://img.shields.io/github/issues/Mr-ZHJ/IR-RTDETR?style=flat-square&color=orange" alt="Issues"></a>
  <br><br>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1.1+-EE4C2C?style=flat-square" alt="PyTorch 2.1.1+">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat-square" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-A42E2B?style=flat-square" alt="AGPL-3.0 License">
  <br><br>
  <img src="https://img.shields.io/badge/mAP50-85.4%25-brightgreen?style=flat-square" alt="mAP50 85.4%">
  <img src="https://img.shields.io/badge/FLOPs-41.3G-success?style=flat-square" alt="FLOPs 41.3G">
  <img src="https://img.shields.io/badge/Params-13.4M-success?style=flat-square" alt="Params 13.4M">
</p>

> [中文版](./README-zh.md) | **English**

An improved RT-DETR model for **infrared small UAV (unmanned aerial vehicle) detection** in complex scenes. IR-RTDETR improves upon RT-DETR-l by introducing **ADown + C2f-FAT** in the backbone for low-loss downsampling and multi-scale window attention, designing a **MSCAS-AIFI** module with additive token mixing to replace the standard MHSA in the encoder, and replacing RepC3 with a **FMA (Fast Mixed Aggregation Network)** in the cross-scale fusion module.

---

## 📑 Table of Contents

- [📝 Overview](#-overview)
- [✨ Key Improvements](#-key-improvements)
- [🏗️ Architecture](#-architecture)
- [📁 Project Structure](#-project-structure)
- [📊 Ablation Study](#-ablation-study)
- [🚀 Results](#-results)
- [🔧 Quick Start](#-quick-start)
  - [⚙️ Installation](#-installation)
  - [🔥 Training](#-training)
  - [🔍 Inference](#-inference)
  - [✅ Evaluation](#-evaluation)
- [📋 Key Configuration](#-key-configuration)
- [🧩 Custom Modules](#-custom-modules)
- [🗃️ Datasets](#-datasets)
- [🙏 Acknowledgments](#-acknowledgments)
- [📜 License](#-license)

---

## 📝 Overview

> 📌 **Note**: This project is **Part II (Chapter 3)** of the master's thesis *"Research on Small UAV Target Detection Algorithms in Complex Scenes"*. 

Infrared small UAV detection suffers from extremely small pixel size, weak features, and strong background clutter — making them exceptionally difficult to detect with standard detectors. IR-RTDETR addresses these issues through three key innovations:

1. **ADown + C2f-FAT Backbone** — Split heterogeneous downsampling (avg-pool + max-pool dual-branch) preserves weak infrared target features, while FAT_Block (FDWA + FMFFN) enhances small target saliency through multi-scale window attention and frequency modulation.
2. **MSCAS-AIFI Encoder** — Multi-scale convolutional additive self-attention replaces the original AIFI's matrix-multiplication MHSA, reducing complexity from O(N²) to O(N) while enhancing global token interaction for small targets.
3. **FMA Fusion Module** — A 4-branch feature decoupling design with partial convolution (FastNet) replaces RepC3 in CCFM, achieving lightweight yet discriminative cross-scale feature fusion.

---

## ✨ Key Improvements

| Component | Original RT-DETR | Proposed | Benefit |
|-----------|-----------------|----------|---------|
| Backbone downsampling | Standard Conv (stride=2) | **ADown**: split heterogeneous downsampling | Weakens background clutter, preserves small target edge/texture features |
| Backbone feature extraction | C2f | **C2f-FAT**: C2f + FAT_Block (FDWA + FMFFN) | Multi-scale window attention focuses on small target regions; frequency modulation suppresses background low-frequency noise |
| Encoder attention (AIFI) | Multi-Head Self-Attention (MHSA, O(N²)) | **MSCAS-AIFI**: Multi-Scale Convolutional Additive Self-Attention (O(N)) | Additive token mixing replaces matrix multiplication; MSSA + ECA dual enhancement on Q/K; DropPath for generalization |
| CCFM fusion block | RepC3 | **FMA (MANet_FasterBlock)**: 4-branch decoupling + partial conv | Channel-level + spatial-texture + raw-passthrough branches; FastNet partial conv reduces computation while preventing noise diffusion |

---

## 🏗️ Architecture

![IR-RTDETR overall framework](images/framework.jpg)

---

## 📁 Project Structure

```
RTDETR/
├── train.py                          # Training entry (FMA1.yaml + uav410sub01.yaml)
├── val.py                            # Evaluation: outputs P/R/mAP/GFLOPs/Params/FPS table
├── detect.py                         # Inference visualization
├── export.py                         # ONNX export
├── track.py                          # Video tracking
├── get_FPS.py                        # FPS/latency benchmarking
├── get_COCO_metrice.py               # COCO metrics + TIDE error analysis
├── get_model_erf.py                  # Effective Receptive Field (ERF) heatmap
├── heatmap.py                        # GradCAM / GradCAM++ / LayerCAM visualization
├── main_profile.py                   # Layer-wise profiling + fuse
├── plot_result.py                    # Training curve plotting from results.csv
├── requirements.txt
├── setup.py / setup.cfg
├── dataset/
│   ├── IRSTD-1K.yaml                 # IRSTD-1K infrared small target dataset
│   ├── uav410sub01.yaml              # Anti-UAV410 infrared UAV dataset (main)
│   ├── split_data.py                 # Train/val/test split tool
│   ├── xml2txt.py                    # XML → YOLO TXT conversion
│   └── yolo2coco.py                  # YOLO → COCO format conversion
├── images/
│   └── framework.jpg                 # Model framework diagram
├── ultralytics/
│   ├── cfg/models/hw-detr/FMA1.yaml  # ★ Final model configuration
│   ├── nn/
│   │   ├── modules/                  # Official ultralytics modules (unmodified)
│   │   └── extra_modules/           # ★ Custom improvements (4 files only)
│   │       ├── block.py             #   ADown, MANet, MANet_FasterBlock, C2f_FAT, Faster_Block
│   │       ├── transformer.py       #   AIFI_CAS_att3, AdditiveTokenMixer_att3
│   │       ├── filc.py              #   FAT_Block (FDWA + FMFFN)
│   │       └── __init__.py
│   ├── models/rtdetr/               # RT-DETR train/val classes
│   └── utils/                         # Framework utilities
└── .gitignore
```

---

## 📊 Ablation Study

Baseline: **RTDETR-R18**, Dataset: **Anti-UAV410**

| Model | ADown+C2f-FAT | MSCAS-AIFI | FMA | P (%) | R (%) | mAP50 (%) | mAP50:95 (%) | GFLOPs | Params (M) |
|-------|:---:|:---:|:---:|:-----:|:-----:|:---------:|:------------:|:------:|:----------:|
| R (baseline) | | | | 91.0 | 79.2 | 81.4 | 43.79 | 56.9 | 19.87 |
| R-A | ✓ | | | 92.2 | 79.6 | 82.7 | 43.85 | 45.7 | 13.39 |
| R-A-M | ✓ | ✓ | | 95.7 | 80.8 | 84.3 | 44.52 | 45.8 | 13.47 |
| **R-A-M-F (Ours)** | ✓ | ✓ | ✓ | **94.4** | **81.5** | **85.4** | **45.17** | **41.3** | **13.41** |

> **R**: RTDETR-R18 . **A**: ADown + C2f-FAT . **M**: MSCAS-AIFI . **F**: FMA

Compared to baseline RTDETR-R18: **mAP50:95 +1.38%, Precision +3.40%, Recall +2.32%**, while **GFLOPs −27.4%, Params −32.5%**.

---

## 🚀 Results

Performance of the proposed model on three infrared datasets:

| Dataset | P (%) | R (%) | mAP50 (%) | APsmall (%) |
|---------|:-----:|:-----:|:---------:|:-----------:|
| Anti-UAV410 | 94.4 | 81.5 | 85.4 | 39.3 |
| Anti-MUAV15 | 97.6 | 96.9 | 97.2 | 63.2 |
| IRSTD-1K | 88.6 | 81.5 | 86.2 | 39.8 |

> Model complexity (identical across datasets): **41.3 GFLOPs**, **13.4M parameters**.

---

## 🔧 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mr-ZHJ/IR-RTDETR.git
cd IR-RTDETR

# 2. Create a Python 3.10+ environment (recommended)
conda create -n irdetr python=3.10 -y
conda activate irdetr

# 3. Install PyTorch (CUDA 11.8)
pip install torch==2.1.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install core dependencies
pip install -r requirements.txt

# 5. Install additional packages
pip install timm==1.0.7 einops grad-cam==1.5.4 \
    dill==0.3.8 albumentations==1.4.11 tidecv prettytable
```

### 🔥 Training

Train IR-RTDETR on the Anti-UAV410 dataset:

```bash
python train.py
```

Or use the ultralytics CLI:

```bash
yolo train model=ultralytics/cfg/models/hw-detr/FMA1.yaml data=dataset/uav410sub01.yaml \
    imgsz=640 epochs=100 batch=8 rect=True workers=4
```

**Key training parameters (as used in the paper):**

The model is trained using the **AdamW** optimizer with an initial learning rate of **0.0001**. We use a batch size of **8** and an input image size of **640×640**. The training process runs for **100** epochs with **rect** training enabled.

### 🔍 Inference

```bash
python detect.py
```

### ✅ Evaluation

```bash
python val.py
```

---

## 📋 Key Configuration

The final model configuration is in `ultralytics/cfg/models/hw-detr/FMA1.yaml`:

```yaml
# Backbone: ADown + C2f-FAT
backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # P2/4
  - [-1, 1, C2f, [128]]
  - [-1, 1, ADown, [256]]           # P3/8  ← Improved downsampling
  - [-1, 1, C2f_FAT, [256]]         # ← FAT (FDWA + FMFFN)
  - [-1, 1, ADown, [384]]           # P4/16
  - [-1, 1, C2f_FAT, [384]]
  - [-1, 1, ADown, [384]]           # P5/32
  - [-1, 3, C2f, [384]]

# Head: MSCAS-AIFI + FMA (MANet_FasterBlock)
head:
  - [-1, 1, AIFI_CAS_att3, [1024, 8]]   # ← Additive self-attention
  - [-1, 1, MANet_FasterBlock, [256]]   # ← FMA fusion (×5)
  - [[20, 23, 26], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]
```

---

## 🧩 Custom Modules

All custom improvements are in `ultralytics/nn/extra_modules/`:

| Module | File | Class | Description |
|--------|------|------|-------------|
| ADown | `block.py` | `ADown` | Split heterogeneous downsampling |
| C2f-FAT | `block.py` | `C2f_FAT` | C2f + FAT_Block (window attention + freq modulation) |
| FAT_Block | `filc.py` | `FAT_Block` | FDWA + FMFFN feature attention block |
| MSCAS-AIFI | `transformer.py` | `AIFI_CAS_att3` | Multi-scale conv additive self-attention |
| AdditiveTokenMixer | `transformer.py` | `AdditiveTokenMixer_att3` | O(N) additive token mixing |
| FMA | `block.py` | `MANet_FasterBlock` | 4-branch aggregation + partial conv |

---

## 🗃️ Datasets

| Dataset | Type | Images | Target Size | Source |
|---------|------|--------|-------------|--------|
| **Anti-UAV410** | Single-target IR UAV | 2500 (subset) | Avg 1196 px² (0.36%) | [Anti-UAV](https://github.com/flying-violin/anti-uav) |
| **Anti-MUAV15** | Multi-target IR UAV | — | Avg 97 px² (0.03%) | [Anti-MUAV](https://github.com/flying-violin/Anti-MUAV) |
| **IRSTD-1K** | General IR small target | 1000 | Avg 83 px² (0.03%) | [IRSTD-1K](https://github.com/RuiZhang2016/IRSTD-1K) |

---

## 🙏 Acknowledgments

This project builds upon the following open-source works:

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) — [CVPR 2024] Real-Time Detection Transformer
- [Ultralytics](https://github.com/ultralytics/ultralytics) — Framework and YOLO ecosystem
- [Anti-UAV](https://github.com/flying-violin/anti-uav) — Infrared UAV tracking dataset
- [IRSTD-1K](https://github.com/RuiZhang2016/IRSTD-1K) — Infrared small target dataset

---

## 📜 License

This project is licensed under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), inherited from the Ultralytics framework.
