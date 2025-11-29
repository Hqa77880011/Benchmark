
Project Overview

This project includes implementations of object detection and image segmentation based on deep learning models such as YOLO, SAM, and Swin-UNet (Transformer). We provide complete datasets and pre-trained models to help you quickly test or retrain the system.

Project Structure

├── data.zip                # Dataset archive (needs to be extracted)

├── yolo/                   # YOLO model and weight files

├── sam/                    # SAM (Segment Anything Model) and scripts

├── unet/                   # Swin-UNet Transformer-based segmentation model

├── compute.py              # Evaluation metrics calculation script

└── README.md               # Project documentation

Data Information

All data are stored in data.zip. Please extract the file before use:

unzip data.zip -d ./data

After extraction, the structure becomes:

./data/

├── images/     # Raw images

├── masks/      # Segmentation masks

Model Description

Pre-trained model weights are provided in the following folders:

Model Type    Path     Description

YOLO          yolo/    Object detection model supporting multi-class detection

SAM           sam/     Segmentation model that can generate masks for arbitrary objects

Swin-UNet     unet/    Transformer-based image segmentation model

You may directly test these models on your images or retrain them using our dataset.

Quick Start

1\. Use Pre-trained Models

Example: YOLO detection

python test.py

Example: Swin-UNet segmentation

python test.py

Example: SAM segmentation

python helpers/extract\_embeddings.py --checkpoint-path sam\_vit\_h\_4b8939.pth --dataset-folder data

python helpers/generate\_onnx.py --checkpoint-path sam\_vit\_h\_4b8939.pth --onnx-model-path ./sam\_onnx.onnx --orig-im-size 360 360

2\. Train Your Own Models

Example: Train YOLO

python yolo/trains.py

Example: Train Swin-UNet

python train.py --output\_dir ./model\_out/datasets --dataset datasets --img\_size 224 --batch\_size 32 --cfg configs/swin\_tiny\_patch4\_window7\_224\_lite.yaml --root\_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET\_OUTPUT/nnunet\_preprocessed/Dataset001\_mm/nnUNetPlans\_2d\_split

Model Evaluation

We provide compute.py to calculate common segmentation metrics:

\- IoU

\- Dice coefficient

\- Precision

\- Recall

Example:

python compute.py --pred ./results/masks --gt ./data/masks

Project Features

✓ Supports YOLO / SAM / Transformer architectures  

✓ Includes pre-trained model weights  

✓ Provides reproducible datasets  

✓ Suitable for training and validating on custom data  


