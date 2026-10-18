# LithoDefectV1

Implementation of **Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation**.

The repository contains synthetic SEM-style image/mask pairs, synthesis tools, YOLO and Swin-UNet training/inference, and an image-prompted SAM baseline.

## Installation

Install Git LFS, create and activate a Python environment, then run from the repository root:

```bash
git lfs install --local
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision
python -m pip install -r synthesis_pipeline/requirements.txt
python -m pip install -r unet/requirements.txt -e ./yolo
```

In PowerShell, use `.venv\Scripts\Activate.ps1` for activation. Select matching PyTorch/torchvision builds for your device.

## Data and preprocessing

Acquire the repository's LFS `data.zip`, extract paired PNGs and export both model formats:

```bash
python prepare_inputs.py --plan
python prepare_inputs.py --source data --output output/benchmark --seed 42
```

Use empty source/output directories. `--offline --archive path/to/data.zip` accepts an already downloaded archive. The release contains synthetic image/mask pairs across eight categories. Preparation keeps `_aug_N` derivatives of one original together, saves the seed/group assignments in `splits.json`, and writes `yolo/` and binary `swin_unet/` train/val/test inputs.

To synthesize additional data, supply the background and defect inputs named in `synthesis_pipeline/config.yaml`, set its seed/output paths, then run:

```bash
python synthesis_pipeline/run_pipeline.py --list-steps
python synthesis_pipeline/run_pipeline.py --config synthesis_pipeline/config.yaml
```

Private/public backgrounds in this generator are synthetic source roles; they do not denote acquired fab images. The saved source-grouped utility split is the current executable protocol. [Protocol and checkpoint notes](docs/REPRODUCIBILITY.md) describe the input schemas and historical-resource limits.

## Training and inference

YOLO segmentation (the pretrained model downloads on first use; `yolo11n-seg.yaml` selects training from scratch):

```bash
python yolo/trains.py --data output/benchmark/yolo/data.yaml --model yolo11n-seg.pt --device 0 --imgsz 384 --epochs 50 --batch 64 --seed 42 --project runs/segment --name lithodefect
python yolo/predict_masks.py --model runs/segment/lithodefect/weights/best.pt --source output/benchmark/yolo/images/test --output output/yolo_predictions --device cpu --imgsz 384 --conf 0.25
```

Ultralytics increments existing run names; use the checkpoint path printed by training. Swin-UNet binary segmentation:

```bash
git lfs pull --include="unet/pretrained_ckpt/*.pth"
python unet/train.py --root_path output/benchmark/swin_unet --dataset LithoDefect --output_dir output/swin_model --num_classes 2 --img_size 224 --batch_size 32 --max_epochs 50 --seed 42
python unet/test.py --root_path output/benchmark/swin_unet --dataset LithoDefect --checkpoint output/swin_model/best_model.pth --output_dir output/swin_model --test_save_dir output/swin_predictions --num_classes 2 --split_name test --device cpu
```

Image-only contrast-prompted SAM uses the matching upstream checkpoint:

```bash
python -m pip install -e sam/segment-anything scipy Pillow
python sam/download_checkpoint.py --model-type vit_b --output checkpoints/sam_vit_b.pth
python sam/predict_masks.py --images output/benchmark/swin_unet/test/img --output output/sam --checkpoint checkpoints/sam_vit_b.pth --model-type vit_b
```

The SAM predictor does not read masks. It writes one original-size binary mask per image and records cohort/prompt settings in `protocol.json`. Choose alternate thresholds on validation inputs before test prediction.

## Evaluation

Evaluate the complete paired test cohort with the same category averaging:

```bash
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/yolo_predictions --output-dir output/yolo_metrics --suffix= --average category
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/swin_predictions --output-dir output/swin_metrics --suffix= --average category
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/sam --output-dir output/sam_metrics --suffix= --average category
```

Missing predictions raise an error; image-size alignment is opt-in. Scores are generated from supplied masks. No manuscript performance values are presented as repository run results.

## Outputs

- `output/benchmark/splits.json`: seed and source-group split assignments.
- `output/benchmark/{splits,yolo,swin_unet}/`: paired and model-ready datasets.
- YOLO run folder and `output/swin_model/`: trained checkpoints and training records.
- `output/{yolo_predictions,swin_predictions,sam}/`: binary prediction PNGs; SAM additionally writes `protocol.json`.
- Evaluation folders: per-image/category metrics and aggregate summaries written by `compute.py`.

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, PyTorch 2.7.1+cu118 and torchvision 0.22.1 (installed versions inspected). This update checked relevant Python syntax, YAML/configuration and CLI definitions. Training, full evaluation and CI were not run.

Bundled model packages retain their upstream licenses and citations: [Ultralytics](yolo/LICENSE), [Swin-Unet](unet/UPSTREAM.md), and [SAM](sam/segment-anything/LICENSE).
