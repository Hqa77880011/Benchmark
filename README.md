本项目包含基于 YOLO、SAM、Swin-UNet (Transformer) 等深度学习模型的目标检测与图像分割实现。我们提供了完整的数据与训练好的模型，方便你快速测试或二次训练。

📂 项目结构

├── data.zip # 数据集压缩包（需解压）

├── yolo/ # YOLO 模型及权重文件

├── sam/ # SAM (Segment Anything Model) 模型及脚本

├── unet/ # Swin-UNet 结构模型及相关代码

├── compute.py # 计算指标的代码

└── README.md # 项目说明文件

📦 数据说明

所有数据存放在 data.zip 文件中。请先解压该文件：



unzip data.zip -d ./data

解压后目录结构如下：



./data/

├── images/ # 原始图像

├── masks/ # 分割标签

🧩 模型说明

我们在以下文件夹中提供了训练好的模型权重：

| 模型类型 | 文件夹路径 | 说明  |
| --- | --- | --- |
| YOLO | yolo/ | 目标检测模型，支持多类别检测 |
| SAM | sam/ | 分割模型，可对任意物体生成 mask |
| Swin-UNet | unet/ | 基于 Transformer 的视觉分割模型 |

你可以直接使用这些模型在你的数据上进行测试，也可以基于我们提供的数据进行重新训练。

🚀 快速开始

1️⃣ 使用预训练模型进行测试


\# 示例：使用 YOLO 模型检测图像

python test.py

\# 示例：使用 Swin-unet 模型分割图像

python test.py

\# 示例：使用 SAM 模型分割图像

python helpers/extract_embeddings.py --checkpoint-path sam_vit_h_4b8939.pth --dataset-folder data 

python helpers/generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 360 360

2️⃣ 使用数据训练你自己的模型

bash

\# 示例：训练 YOLO 模型

python yolo/trains.py

\# 示例：训练 Swin-unet 模型

python train.py --output_dir ./model_out/datasets --dataset datasets --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split

📈 模型评估

我们提供了一个评估脚本 compute.py，可计算以下指标：

IoU 

Dice 系数

Precision

Recall

示例：



python compute.py --pred ./results/masks --gt ./data/masks

💡 项目特点

✅ 支持 YOLO / SAM / Transformer 三种模型架构

✅ 提供 预训练模型权重

✅ 提供 可复现的数据集

✅ 适合 自定义数据的训练与验证