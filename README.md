# LithoDefectV1 — 光刻缺陷分割合成基准

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgreen.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation**
>
> Qinao Hu, Jiwei Shen — East China Normal University
>
> *IEEE Transactions on Semiconductor Manufacturing, 2025 (Under Review)*

---

## 项目简介

在半导体制造的光刻工艺中，缺陷检测是良率控制的关键环节。然而，真实的晶圆缺陷数据受到严格的工业保密限制，导致学术界的缺陷检测研究长期面临"数据孤岛"困境——研究者无法获取公开可复现的数据集。

**LithoDefectV1** 是首个面向光刻缺陷分割的公开基准数据集，提供：

- **3600 张**高分辨率合成 SEM（扫描电子显微镜）风格图像
- **像素级**精确标注（pixel-perfect annotation）
- 覆盖 **8 类**典型光刻缺陷
- 配套的**数据合成流水线**、**模型训练代码**和**评估工具**

核心思路：用**隐私保护的合成流水线**从少量种子缺陷生成大规模多样化数据，绕过工业保密限制，为自动化缺陷检测提供可复现的研究基准。

---

## 项目结构

```
Benchmark/
├── data.zip                       # 数据集压缩包（需解压使用）
├── compute.py                     # 分割评估指标计算（IoU / Dice / Precision / Recall）
├── README.md                      # 项目文档（本文件）
│
├── synthesis_pipeline/            # 【核心】数据合成流水线
│   ├── run_pipeline.py            #   一键运行入口
│   ├── config.yaml                #   全局配置文件
│   ├── lithodefect/               #   Python 工具包
│   │   ├── synthesis.py           #     合成引擎（主控）
│   │   ├── augment.py             #     几何/光度增强
│   │   ├── transforms.py          #     空间变换（仿射/透视/桶形畸变）
│   │   ├── composite.py           #     图像合成（羽化 + Alpha 混合）
│   │   ├── split.py               #     数据集分层划分
│   │   ├── export_yolo.py         #     导出 YOLO 训练格式
│   │   ├── export_swinunet.py     #     导出 Swin-UNet 训练格式
│   │   └── dataset.py             #     PyTorch Dataset 接口
│   └── tools/                     #   辅助工具
│       ├── expand_data.py         #     数据集扩充（3× 扩展）
│       └── convert_labels.py      #     标注格式转换
│
├── yolo/                          # YOLO 实例分割模型
│   ├── trains.py                  #   训练脚本（支持 YOLOv8/v9/v11/v12）
│   ├── data.yaml                  #   数据集配置
│   └── best_8 ~ best_12           #   预训练权重文件
│
├── unet/                          # Swin-UNet Transformer 分割模型
│   ├── train.py                   #   训练脚本
│   ├── test.py                    #   测试/推理脚本
│   ├── trainer.py                 #   训练器封装
│   ├── config.py                  #   模型配置
│   ├── utils.py                   #   工具函数
│   ├── configs/                   #   模型结构 YAML 配置
│   ├── networks/                  #   网络结构定义（Swin Transformer + U-Net）
│   ├── datasets/                  #   数据加载器（Synapse 格式）
│   ├── pretrained_ckpt/           #   预训练权重（ImageNet）
│   └── best_model.pth             #   在 LithoDefectV1 上的最佳模型
│
└── sam/                           # SAM 零样本分割
    └── SAM-Tool/                  #   基于 SAM 的交互式标注工具
        ├── segment_anything_annotator.py  # 标注主程序
        └── cocoviewer.py                  # COCO 标注可视化
```

---

## 数据集信息

### 解压数据

```bash
unzip data.zip -d ./data
```

### 解压后结构

```
data/
├── images/                        # 原始图像（360×360 px）
├── masks/                         # 分割掩码（与图像一一对应）
├── defects/                       # 种子缺陷素材
│   ├── images/                    #   缺陷抠图（arching / particle / peeling / residue）
│   └── masks/                     #   对应二值掩码
└── backgrounds/                   # 背景图像（256×256 px）
    ├── shen_*.png                 #   公开背景（用于 test_public 集）
    ├── chip_*.png                 #   私有背景（用于 train/val/test_private 集）
    └── luhu_*.png                 #   私有背景
```

### 缺陷类别一览（8 类）

| 编号 | 类别名 | 缺陷类型 | 样本数 | 描述 |
|:---:|:---|:---|:---:|:---|
| 0 | **ArchDeform** | 几何变形 | 500 | 图案扭曲（OPC 失效、光刻胶倒塌） |
| 1 | **Block** | 高深宽比 | 100 | 块状线端缩短 |
| 2 | **Droplet** | 异物残留 | 350 | 液滴状残留物 |
| 3 | **Flake** | 异物残留 | 350 | 片状碎屑 |
| 4 | **Needle** | 高深宽比 | 450 | 针状桥接缺陷 |
| 5 | **ParticleContam** | 异物污染 | 400 | 颗粒污染物 |
| 6 | **ResidueLeft** | 工艺残留 | 1000 | 不完全刻蚀/显影残留 |
| 7 | **Spindle** | 几何变形 | 450 | 纺锤形桥接缺陷 |
| | | **合计** | **3600** | |

### 数据集划分（分层采样，随机种子 42）

| 子集 | 比例 | 背景来源 | 用途 |
|:---|:---:|:---|:---|
| train | 70% | 私有背景 | 模型训练 |
| val | 10% | 私有背景 | 验证调参 |
| test_private | 10% | 私有背景 | 闭集测试（同分布） |
| test_public | 10% | 公开背景 | 开集测试（跨域泛化） |

---

## 合成流水线

### 流程图

```
┌──────────────────────┐
│  种子缺陷图像 + 掩码  │  ← 从有限真实样本中提取
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 1: 几何增强   │  随机翻转、旋转、缩放
│                      │  弹性变形、透视变形、桶形畸变
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 2: 图像合成   │  高斯羽化 + Alpha 混合
│                      │  贴到公开/私有背景图案上
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  3600 对合成图像-掩码 │ 360×360 px，像素级精确标注
└──────────────────────┘
```

### 关键设计

- **避免中心偏置**：缺陷不会放置在图像正中央（`avoid_center: true`），防止模型学到"缺陷总是在中间"的虚假先验
- **双背景组策略**：私有背景用于训练集，公开背景用于测试集——模型必须在训练时从未见过的背景上做推理，才能检验真实泛化能力
- **随机化参数**：每张图像的缩放比例（0.7~1.3）、透视偏移（±10%）、桶形畸变系数（k1∈[-0.3, 0.3]）均独立随机采样

详细说明参见 `synthesis_pipeline/README.md`。

---

## 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10（模型训练需要 GPU；纯合成流水线 CPU 即可）

### 1. 安装依赖

```bash
# 合成流水线
pip install -r synthesis_pipeline/requirements.txt

# Swin-UNet 训练（需要 GPU）
pip install -r unet/requirements.txt

# YOLO 训练
pip install ultralytics
```

### 2. 解压数据集

```bash
unzip data.zip -d ./data
```

### 3. 运行合成流水线（从头生成数据）

```bash
cd synthesis_pipeline

# 编辑 config.yaml 配置路径和参数

# 一键运行全流程（合成 → 划分 → 导出）
python run_pipeline.py

# 或分步运行
python run_pipeline.py --step synthesis    # 仅合成
python run_pipeline.py --step split       # 仅划分
python run_pipeline.py --step yolo        # 导出 YOLO 格式
python run_pipeline.py --step swinunet    # 导出 Swin-UNet 格式
```

### 4. 训练模型

**YOLO 实例分割：**

```bash
cd yolo
# 编辑 trains.py 中的 data_yaml_path 和预训练模型路径
python trains.py
```

**Swin-UNet 语义分割：**

```bash
cd unet
python train.py \
    --output_dir ./model_out \
    --dataset datasets \
    --img_size 224 \
    --batch_size 32 \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path <数据路径>
```

### 5. 测试/推理

```bash
# YOLO 推理
cd yolo
yolo predict model=best.pt source=<图片路径>

# Swin-UNet 推理
cd unet
python test.py --model best_model.pth --input <输入路径> --output <输出路径>
```

### 6. 评估指标

```bash
python compute.py \
    --gt-dir ./data/masks \
    --pred-dir ./results/pred_masks \
    --output-dir ./results \
    --suffix _mask
```

输出的 `metrics_results.txt` 包含每张图像及整体的 IoU / Dice / Precision / Recall。

---

## 基准结果

以下结果来自论文中的对比实验（Open-test = test_public 集，360×360 输入分辨率）：

| 模型 | 测试集 | mIoU | Dice | Precision | Recall | 备注 |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| YOLOv8 | Open-test | 0.851 | 0.918 | 0.866 | 0.979 | 单阶段实例分割 |
| YOLOv9 | Open-test | 0.855 | 0.920 | 0.872 | 0.977 | PGI + GELAN 架构 |
| YOLOv11 | Open-test | 0.853 | 0.919 | 0.869 | 0.978 | Ultralytics 最新版 |
| YOLOv12 | Open-test | 0.849 | 0.917 | 0.863 | 0.981 | 基于注意力机制 |
| SAM (zero-shot) | Open-test | 0.788 | 0.880 | **0.998** | 0.790 | 零样本，不做训练 |
| **Swin-UNet** | Open-test | **0.873** | **0.927** | 0.934 | 0.926 | 🏆 Transformer 语义分割 |

### 关键发现

1. **Swin-UNet 精度最优**：Transformer 的全局自注意力机制捕获了光刻缺陷的长程形态特征，mIoU 和 Dice 均为最高
2. **YOLO 系列速度最快**：在精度仅略低（~2% mIoU）的情况下，推理速度远超 Transformer 方案，适合在线产线部署
3. **SAM 零样本的启示**：Precision 高达 0.998（几乎不误检），但 Recall 仅 0.790（漏检严重），说明通用大模型直接用于专业领域仍需适配

---

## Python API 使用

合成流水线支持 Python 编程方式调用：

```python
from lithodefect.synthesis import Synthesizer
from lithodefect.split import split_dataset
from lithodefect.export_yolo import export_to_yolo
from lithodefect.export_swinunet import export_to_swinunet
from lithodefect.dataset import LithoDefectDataset

# Step 1: 合成
syn = Synthesizer({...})  # 配置字典同 config.yaml
syn.run()

# Step 2: 划分
split_dataset("output/synthetic_defects", "output/split", seed=42)

# Step 3: 导出
export_to_yolo("output/split", "output/yolo", class_mapping={...})
export_to_swinunet("output/split", "output/swin_unet", class_mapping={...})

# Step 4: 在 PyTorch 训练中使用
dataset = LithoDefectDataset("output/swin_unet/train", mode='train')
img, mask = dataset[0]  # img: (3, H, W) Tensor, mask: (H, W) Tensor
```

---

## 引用

如果您在研究中使用了 LithoDefectV1 数据集或合成流水线，请引用：

```bibtex
@article{hu2025lithodefect,
  title    = {Advancing Lithography Process Control in Integrated Circuit
              Manufacturing via a Synthetic Benchmark for Defect Segmentation},
  author   = {Hu, Qinao and Shen, Jiwei},
  journal  = {IEEE Transactions on Semiconductor Manufacturing},
  year     = {2025},
  note     = {Under review}
}
```

---

## 许可证

- 本项目代码采用 [MIT License](https://opensource.org/licenses/MIT)
- LithoDefectV1 数据集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 协议发布

---

## 联系方式

- **代码与数据集问题**：[GitHub Issues](https://github.com/Hqa77880011/Benchmark/issues)
- **邮件**：qinaoH@stu.ecnu.edu.cn
