# LithoDefectV1 — A Synthetic Benchmark for Lithography Defect Segmentation

> **Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation**
>
> Qinao Hu, Jiwei Shen — East China Normal University
>
> **IEEE Transactions on Semiconductor Manufacturing: Major revision completed; under review.**

LithoDefectV1 contains 3,600 pairs of synthetic SEM-style images and pixel-level defect annotations spanning 8 defect categories. This repository provides the synthesis pipeline, YOLO and Swin-UNet training entry points, SAM tools, and mask evaluation scripts.

## Manuscript Revision and Evaluation Protocol

The revised manuscript uses the following three-tier evaluation protocol. Open-test measures in-distribution generalization on held-out samples; it does not represent open-set recognition or cross-background-domain evaluation.

| Split / Scenario | Images | Definition |
| --- | ---: | --- |
| Train | 2,518 | Training samples from LithoDefectV1 |
| Open-test | 1,082 | Synthetic samples held out through a stratified 7:3 split |
| Re-synth-test | 300 | Samples independently generated using the same synthesis pipeline with a different random seed |
| Fab-test | 92 | Real industrial SEM images acquired outside the synthesis pipeline and independently annotated |

SAM is evaluated in a zero-shot setting on a **stratified subset of 129 Open-test images** and the complete Re-synth-test and Fab-test sets. Its Open-test sample count differs from that of the other models.

This update fixes synthesis parameter dispatch, LFS pointers in source files, validation data loading, data pairing, and execution entry points. See the [reproducibility notes](docs/REPRODUCIBILITY.md) for version boundaries and validation procedures. The results below are transcribed from the revised manuscript; this repository maintenance did not retrain the models or recompute those experiments.

## Repository Structure

```text
Benchmark/
├── README.md
├── docs/REPRODUCIBILITY.md       # Paper protocol, code behavior, resources, and validation
├── data.zip                      # Dataset resource managed with Git LFS
├── prepare_dataset.py            # Released archive → training-ready splits and exports
├── compute.py                    # Binary mask evaluation with complete prediction pairing
├── tests/test_workflow.py         # Regression checks for the fixes in this update
├── synthesis_pipeline/
│   ├── run_pipeline.py           # Synthesis → splitting → format export
│   ├── config.yaml
│   └── lithodefect/
├── yolo/                         # Bundled Ultralytics implementation and its license
│   ├── trains.py                 # Parameterized instance segmentation training entry point
│   ├── predict_masks.py          # One binary prediction PNG per input image
│   └── ultralytics/
├── unet/                         # Swin-UNet, PNG/NPZ loaders, training, and inference
│   └── UPSTREAM.md               # Original Swin-Unet documentation and citation
└── sam/                          # SAM implementation and interactive annotation tools
```

Upstream demo websites, examples, documentation builds, test infrastructure, caches, and old logs are excluded from the benchmark's runtime dependencies. Model implementations, configurations, licenses, and existing checkpoint resources are retained.

## Installation

Clone the repository without automatically downloading every LFS object, then enable LFS locally:

```bash
git -c 'filter.lfs.process=git-lfs filter-process --skip' clone https://github.com/Hqa77880011/Benchmark.git
cd Benchmark
git lfs install --local
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. All commands below start from the repository root; no directory changes are required.

```bash
python -X utf8 -m pip install --upgrade pip
python -X utf8 -m pip install torch torchvision
python -X utf8 -m pip install -r synthesis_pipeline/requirements.txt
python -X utf8 -m pip install -r unet/requirements.txt -e ./yolo
```

Install a matching PyTorch/torchvision pair for your platform and device before installing the remaining dependencies. The CPU workflow checks used Python 3.13.5, PyTorch 2.7.1, and torchvision 0.22.1. The UTF-8 option also supports Windows checkouts containing non-ASCII paths. Synthesis and mask evaluation run on CPU; use `--device cpu` for CPU model runs. CUDA training requires compatible CUDA-enabled PyTorch and torchvision builds.

## Prepare the Released Dataset

Download and extract the published image-mask archive:

```bash
python prepare_inputs.py --plan
python prepare_inputs.py --source data --output output/benchmark --seed 42
```

The verified archive contains **3,600 pairs**, arranged as follows:

```text
data/
├── img/<category>/<Category>_...png
└── mask/<category>/<Category>_...png
```

Images are RGB, masks are grayscale, and paired files have identical names. Both have a resolution of 360 × 360 pixels. The archive contains final benchmark pairs, not the seed defects/backgrounds required for synthesis.

The command acquires LFS data when absent, checks paired archive members, safely extracts PNGs, and exports both model formats. If the archive was already extracted, prepare it directly:

```bash
python prepare_dataset.py --source data --output output/benchmark --seed 42
```

The current split groups filenames after removing `_aug_N` within each category.
All 50 augmented variants of one original stay in the same role. Seed 42 and
target ratios 70/10/20 produce **2,350 train / 400 validation / 850 test images**
from **47 / 8 / 17 original groups** in the released archive. Minimum group
counts take priority over exact ratios. Block has only two originals: one train
and one test, with no Block validation examples. All eight categories remain in
train and test. `splits.json` records original-group IDs, per-category counts and
every image path. Use empty source/output directories for `prepare_inputs.py`,
or an empty output with already-extracted data for `prepare_dataset.py`.
`--offline` on the acquisition script requires a local complete archive.

```text
output/benchmark/
├── splits/{train,val,test}/{img,mask}/<category>/*.png
├── splits.json
├── yolo/
│   ├── data.yaml
│   ├── images/{train,val,test}/*.png
│   └── labels/{train,val,test}/*.txt
└── swin_unet/{train,val,test}/{img,mask}/*.png
```

Swin-UNet masks use background=0 and defect=1 by default, with source values greater than 127 treated as defect. YOLO polygon export uses Otsu thresholding, a minimum contour area of 50 pixels, and polygon approximation factor 0.002. For a separate background-plus-eight-category experiment, prepare with `--multiclass` and train with `--num_classes 9`.

## Train and Evaluate YOLO

Use the bundled Ultralytics implementation installed above:

```bash
python yolo/trains.py --data output/benchmark/yolo/data.yaml --model yolo11n-seg.pt --device 0 --imgsz 384 --epochs 50 --batch 64 --project runs/segment --name lithodefect
```

`--model` must identify a segmentation checkpoint or architecture. The example may download the official pretrained checkpoint. To initialize from scratch, use `--model yolo11n-seg.yaml`. For a small CPU run, reduce the batch size and epoch count and pass `--device cpu --workers 0`. If an output directory already exists, Ultralytics increments its run name; use the checkpoint path printed by the training command.

Export a binary mask for every test image and evaluate against the prepared binary ground truth:

```bash
python yolo/predict_masks.py --model runs/segment/lithodefect/weights/best.pt --source output/benchmark/yolo/images/test --output output/yolo_predictions --device cpu --imgsz 384 --conf 0.25
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/yolo_predictions --output-dir output/yolo_metrics --suffix= --average category
```

The exporter merges predicted instances, restores the original image resolution, and writes an all-zero mask when no defect is detected. Use an empty prediction directory so a previous run cannot contribute stale files. Native YOLO validation reports detection/instance metrics; `compute.py` reports foreground mask metrics, so those outputs are distinct.

## Train and Evaluate Swin-UNet

Fetch the pretrained Swin-T encoder before training:

```bash
git lfs pull --include="unet/pretrained_ckpt/*.pth"
python unet/train.py --root_path output/benchmark/swin_unet --dataset LithoDefect --output_dir output/swin_model --num_classes 2 --img_size 224 --batch_size 32 --max_epochs 50
```

The checkpoint path in the Swin-UNet YAML is resolved relative to that YAML file. PNG input reads the existing train/val directories without re-splitting; validation uses deterministic preprocessing. For a CPU run, add `--device cpu --num_workers 0` and reduce the batch size. Legacy NPZ inputs still accept `--dataset datasets --list_dir <list_folder>`.

```bash
python unet/test.py --root_path output/benchmark/swin_unet --dataset LithoDefect --checkpoint output/swin_model/best_model.pth --output_dir output/swin_model --test_save_dir output/swin_predictions --num_classes 2 --split_name test --device cpu
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/swin_predictions --output-dir output/swin_metrics --suffix= --average category
```

`--test_save_dir` controls the class-index PNG output directory. The test script's auxiliary Dice/HD95 log uses a different empty-mask convention from `compute.py`; use the latter for the four foreground mask metrics shown in this repository.

## Existing Checkpoints

Download existing trained checkpoints only when needed:

```bash
git lfs pull --include="unet/best_model.pth,yolo/best_*"
```

The published `unet/best_model.pth` has **4 output channels**, so loading it requires `--num_classes 4`, not the 2-channel setting used for new binary training above. The original four-channel label mapping is not supplied. The existing YOLO files have no filename extension; copy a selected file to a `.pt` filename before passing it to Ultralytics:

```bash
python -c "from shutil import copyfile; copyfile('yolo/best_11', 'yolo/best_11.pt')"
```

The original sample-level split indices are unavailable. Evaluating historical checkpoints on a newly randomized utility split does not reproduce the paper's held-out protocol.

## Synthesize New Data

Synthesis is a separate workflow for users who have prepared seed defects, masks, and backgrounds as described in the [pipeline documentation](synthesis_pipeline/README.md):

```bash
python synthesis_pipeline/run_pipeline.py --list-steps
python synthesis_pipeline/run_pipeline.py --config synthesis_pipeline/config.yaml
```

Configuration paths are relative to the YAML directory. Default inputs are in `synthesis_pipeline/data/` and outputs in `synthesis_pipeline/output/`. These paths are separate from the extracted `data/` archive and `output/benchmark/` above. Use empty output directories for a fresh run.

The synthesis utility defaults to 70/10/10/10 train, val, private-background synthetic test, and public-background synthetic test splits. Both background sources contribute to train/val. `test_private` is not the real Fab-test set, and `test_public` is not the paper's Open-test. Independently generated Re-synth-test data require a different seed and separate output paths; Fab-test requires independently acquired and annotated real SEM images.

## Evaluation Conventions and SAM

`compute.py` expects prediction names `<GT filename stem><suffix>.png`. The examples use `--suffix=` because the exporters preserve the GT stem. Missing predictions, empty input folders, corrupt images, and size mismatches raise errors. Explicitly pass `--resize` only when nearest-neighbor alignment is intended.

The report records actual filenames, sample count, per-image metrics, and aggregate metrics. `--average image` is the default image macro-average. `--average category` first averages within each `<Category>_` filename group, then gives categories equal weight. Pixel values greater than 0 are foreground; zero-denominator metrics, including two empty masks, are assigned 0.

SAM annotation and ONNX preparation are optional and have additional dependencies. Follow the [bundled SAM tool instructions](sam/SAM-Tool/README.md). Interactive annotation is not the paper's fixed zero-shot evaluation protocol.

## Results Reported in the Revised Manuscript

The following values are transcribed from the revised manuscript and retain four decimal places. SAM uses a subset of 129 images for Open-test.

| Model | Test Set | mIoU | Dice | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8-seg | Open-test | 0.8508 | 0.9177 | 0.8663 | 0.9786 |
| YOLOv8-seg | Re-synth-test | 0.8155 | 0.8841 | 0.8320 | 0.9435 |
| YOLOv8-seg | Fab-test | 0.8130 | 0.8813 | 0.8301 | 0.9422 |
| YOLOv9-seg | Open-test | 0.8552 | 0.9203 | 0.8722 | 0.9770 |
| YOLOv9-seg | Re-synth-test | 0.8202 | 0.8878 | 0.8394 | 0.9420 |
| YOLOv9-seg | Fab-test | 0.8185 | 0.8860 | 0.8375 | 0.9411 |
| YOLOv11-seg | Open-test | 0.8530 | 0.9190 | 0.8687 | 0.9784 |
| YOLOv11-seg | Re-synth-test | 0.8164 | 0.8847 | 0.8336 | 0.9442 |
| YOLOv11-seg | Fab-test | 0.8170 | 0.8845 | 0.8352 | 0.9428 |
| YOLOv12-seg | Open-test | 0.8492 | 0.9168 | 0.8631 | 0.9807 |
| YOLOv12-seg | Re-synth-test | 0.8120 | 0.8820 | 0.8271 | 0.9470 |
| YOLOv12-seg | Fab-test | 0.8144 | 0.8828 | 0.8292 | 0.9453 |
| SAM (zero-shot) | Open-test (129) | 0.7884 | 0.8798 | 0.9978 | 0.7896 |
| SAM (zero-shot) | Re-synth-test | 0.7412 | 0.8365 | 0.9590 | 0.7408 |
| SAM (zero-shot) | Fab-test | 0.7480 | 0.8398 | 0.9594 | 0.7487 |
| Swin-UNet | Open-test | 0.8725 | 0.9273 | 0.9344 | 0.9260 |
| Swin-UNet | Re-synth-test | 0.8280 | 0.8921 | 0.8905 | 0.8804 |
| Swin-UNet | Fab-test | 0.8348 | 0.8927 | 0.8934 | 0.8849 |

## Citation and Licensing

The manuscript remains under review. Published volume, issue, page numbers, and DOI are therefore not listed.

```bibtex
@unpublished{hu_lithodefect_revision,
  title = {Advancing Lithography Process Control in Integrated Circuit Manufacturing via a Synthetic Benchmark for Defect Segmentation},
  author = {Hu, Qinao and Shen, Jiwei},
  note = {Manuscript under review at IEEE Transactions on Semiconductor Manufacturing; major revision completed}
}
```

The dataset retains the CC BY 4.0 declaration in the original README, and project-authored code retains its MIT declaration. Third-party directories retain their respective licenses: see [yolo/LICENSE](yolo/LICENSE) for YOLO, and [sam/segment-anything/LICENSE](sam/segment-anything/LICENSE) and [sam/SAM-Tool/LICENSE](sam/SAM-Tool/LICENSE) for the SAM components. The repository-level license declaration does not override the licenses of third-party code.

## Fixed replacement evaluation setting

Use the saved seed-42 source-grouped utility split for every model, retraining supervised models on that split. Binary Swin uses 2 classes; original-size predictions from every method feed the same category-macro foreground evaluator. The default SAM setting now evaluates **every image** in the utility test cohort, with image-only local-contrast prompts and fixed confidence/area selection. The checkpoint acquisition entry below supplies the official matching weights; prediction uses image-only prompts.

```bash
python -m pip install -e sam/segment-anything scipy Pillow
python sam/download_checkpoint.py --model-type vit_b --output checkpoints/sam_vit_b.pth
python sam/predict_masks.py --images output/benchmark/swin_unet/test/img --output output/sam --checkpoint checkpoints/sam_vit_b.pth --model-type vit_b
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/sam --output-dir output/sam_metrics --suffix= --average category
```

`sam/download_checkpoint.py --plan` prints the official URL and expected bytes.
The downloader supports the three [official SAM model releases](https://github.com/facebookresearch/segment-anything#model-checkpoints),
uses a `.part` file until transfer completion, and checks the release size.
ViT-B is the selected default (375,042,383 bytes); no SA-1B dataset is required.
Downloaded checkpoints stay under ignored `checkpoints/`.

The registered defaults are median window 15, absolute local-contrast threshold at least 12/255 and the image's 99.5th percentile, at most 32 connected-component peak prompts, SAM score >=0.8, proposal area <=15%, and cumulative foreground <=35%. Ties prefer smaller masks then candidate order. Images with no accepted foreground still receive a zero mask. These settings define a contrast-prompted SAM baseline and replace the unspecified historical proposal rule. `protocol.json` records the entire cohort, settings and accepted prompts. Select alternatives on validation only and freeze them before test; do not tune from test masks. Actual scores must be computed from new outputs, separately from the historical table above.

Please report code or dataset issues through [GitHub Issues](https://github.com/Hqa77880011/Benchmark/issues).
