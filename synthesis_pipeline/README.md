# LithoDefectV1 — Synthetic Benchmark for Lithography Defect Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the synthesis pipeline described in:

> **Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation**
>
> Qinao Hu, Jiwei Shen — East China Normal University

**LithoDefectV1** is the first publicly accessible benchmark dataset for lithography defect segmentation, containing **3,600 high-resolution SEM-style images** with **pixel-level annotations** across **8 defect categories**.

📦 **Dataset & Code**: [https://github.com/Hqa77880011/Benchmark](https://github.com/Hqa77880011/Benchmark)

---

## Overview

Semiconductor lithography defect data is heavily restricted by industrial confidentiality — this "data silo" problem prevents reproducible research in automated defect inspection. This repository provides a **privacy-preserving synthesis pipeline** that generates high-fidelity defect segmentation data without exposing proprietary fab information.

### Pipeline Architecture

```
┌─────────────────────┐
│  Seed Defect Images │  (extracted from limited real samples)
│  + Binary Masks     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 1: Geometric │  Random flips, rotation, scaling,
│  Augmentation       │  elastic/perspective/barrel distortion
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2: Image     │  Gaussian feathering + Alpha blending
│  Composition        │  on public background patterns
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3,600 Synthetic    │  8 defect classes, 360×360 px,
│  Image-Mask Pairs   │  pixel-perfect annotations
└─────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

Place your data in the following structure:

```
data/
├── defects/
│   ├── images/          # Seed defect cutouts (PNG)
│   │   ├── arching/     #   arching_001.png, ...
│   │   ├── particle/    #   particle_001.png, ...
│   │   ├── peeling/     #   peeling_001.png, ...
│   │   └── residue/     #   residue_001.png, ...
│   └── masks/            # Corresponding binary masks
│       ├── arching/      #   arching_001_mask.png, ...
│       ├── particle/
│       ├── peeling/
│       └── residue/
└── backgrounds/          # Defect-free background images (256×256)
    ├── shen_001.png      #   Public/open-source backgrounds (shen_ prefix)
    └── chip_001.png      #   Private/fab backgrounds (chip_, luhu_ prefix)
```

### 3. Edit configuration

All parameters are in a single file: **`config.yaml`**

```yaml
paths:
  defect_img_dir: "data/defects/images"
  defect_mask_dir: "data/defects/masks"
  bg_dir: "data/backgrounds"

bg_groups:
  private:
    num_aug: 50    # 50 synthetic samples per seed defect (private bg)
  public:
    num_aug: 30    # 30 synthetic samples per seed defect (public bg)
```

### 4. Run the pipeline

```bash
# Full pipeline (synthesis → split → export)
python run_pipeline.py

# Single step only
python run_pipeline.py --step synthesis
python run_pipeline.py --step split
python run_pipeline.py --step yolo
python run_pipeline.py --step swinunet

# Use custom config
python run_pipeline.py --config my_config.yaml
```

---

## Output Structure

After running the full pipeline:

```
output/
├── synthetic_defects/              # Raw synthetic data
│   ├── private/img/ArchDeform/...  # Private bg, 50x per defect
│   ├── private/mask/ArchDeform/...
│   ├── public/img/ArchDeform/...   # Public bg, 30x per defect
│   └── public/mask/ArchDeform/...
│
├── synthetic_defects_split/        # Stratified split (7:1:1:1)
│   ├── train/img/ + mask/
│   ├── val/img/ + mask/
│   ├── test/private/img/ + mask/
│   └── test/public/img/ + mask/
│
├── synthetic_yolo/                 # YOLO segmentation format
│   ├── images/{train,val,test_private,test_public}/
│   ├── labels/{train,val,test_private,test_public}/
│   └── data.yaml
│
└── swin_unet_data/                 # Swin-UNet format
    └── {train,val,test_private,test_public}/
        ├── img/  (RGB)
        └── mask/ (class-index, 0=background)
```

---

## Dataset Statistics

| Defect Category | Type | Count | Description |
|:---|---:|---:|:---|
| ArchDeform | Geometric | 500 | Pattern distortion (OPC failure, resist collapse) |
| Spindle | Geometric | 450 | Spindle-shaped bridging |
| ParticleContam | Foreign Material | 400 | Particle contamination |
| Flake | Foreign Material | 350 | Flake-shaped debris |
| Droplet | Foreign Material | 350 | Droplet residue |
| ResidueLeft | Process Residue | 1000 | Incomplete etch/development residue |
| Needle | High-Aspect Ratio | 450 | Needle-shaped bridging |
| Block | High-Aspect Ratio | 100 | Block-shaped line-end shortening |
| **Total** | | **3,600** | |

---

## Python API

You can also use the pipeline programmatically:

```python
from lithodefect.synthesis import Synthesizer
from lithodefect.split import split_dataset
from lithodefect.export_yolo import export_to_yolo
from lithodefect.export_swinunet import export_to_swinunet
from lithodefect.dataset import LithoDefectDataset

# Step 1: Synthesis
syn = Synthesizer({...})
syn.run()

# Step 2: Split
split_dataset("output/synthetic_defects", "output/split")

# Step 3: Export
export_to_yolo("output/split", "output/yolo")
export_to_swinunet("output/split", "output/swin_unet")

# Use in training
dataset = LithoDefectDataset("output/swin_unet/train", mode='train')
img, mask = dataset[0]
```

---

## Advanced Usage

### Expanding an existing dataset

```bash
python tools/expand_data.py    # 3x expansion (config in script)
```

### Advanced training-time augmentation

```python
from lithodefect.augment import augment_single
import cv2

img = cv2.imread("defect.png")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Apply TPS + elastic deformation augmentation
aug_img, aug_mask = augment_single(img, mask)
```

---

## Benchmark Results (from paper)

| Model | Test Set | mIoU | Dice | Precision | Recall |
|:---|---:|---:|---:|---:|---:|
| YOLOv8 | Open-test | 0.851 | 0.918 | 0.866 | 0.979 |
| YOLOv9 | Open-test | 0.855 | 0.920 | 0.872 | 0.977 |
| YOLOv11 | Open-test | 0.853 | 0.919 | 0.869 | 0.978 |
| YOLOv12 | Open-test | 0.849 | 0.917 | 0.863 | 0.981 |
| SAM (zero-shot) | Open-test | 0.788 | 0.880 | **0.998** | 0.790 |
| **Swin-UNet** | Open-test | **0.873** | **0.927** | 0.934 | 0.926 |

---

## Citation

If you use LithoDefectV1 in your research, please cite:

```bibtex
@article{hu2025lithodefect,
  title={Advancing Lithography Process Control in Integrated Circuit
         Manufacturing via a Synthetic Benchmark for Defect Segmentation},
  author={Hu, Qinao and Shen, Jiwei},
  journal={IEEE Transactions on Semiconductor Manufacturing},
  year={2025},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The LithoDefectV1 dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

- **Dataset & Code**: [GitHub Issues](https://github.com/Hqa77880011/Benchmark/issues)
- **Email**: qinaoH@stu.ecnu.edu.cn
