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
├── compute.py                    # Binary mask evaluation with complete prediction pairing
├── tests/test_workflow.py         # Regression checks for the fixes in this update
├── synthesis_pipeline/
│   ├── run_pipeline.py           # Synthesis → splitting → format export
│   ├── config.yaml
│   └── lithodefect/
├── yolo/                         # Bundled Ultralytics implementation and its license
│   ├── trains.py                 # Parameterized instance segmentation training entry point
│   └── ultralytics/
├── unet/                         # Swin-UNet, PNG/NPZ loaders, training, and inference
│   └── UPSTREAM.md               # Original Swin-Unet documentation and citation
└── sam/                          # SAM implementation and interactive annotation tools
```

Upstream demo websites, examples, documentation builds, test infrastructure, caches, and old logs are excluded from the benchmark's runtime dependencies. Model implementations, configurations, licenses, and existing checkpoint resources are retained.

## Obtaining Data and Checkpoints

Clone the repository with Git. GitHub source ZIP downloads may contain only LFS pointers for large files.

```bash
git clone https://github.com/Hqa77880011/Benchmark.git
cd Benchmark
git lfs install
git lfs pull --include="data.zip"
python -m zipfile -e data.zip data
```

Download existing model resources as needed instead of fetching all checkpoints at once:

```bash
git lfs pull --include="unet/*.pth,unet/pretrained_ckpt/*.pth,yolo/best_*"
```

The dataset archive and models remain available through their existing LFS objects. Not all large files were downloaded or verified during this update. Running the synthesis pipeline requires separately prepared seed defect images, corresponding masks, and backgrounds; the released image dataset should not be assumed to contain the complete seed material. See the [synthesis pipeline documentation](synthesis_pipeline/README.md) for the required input structure.

## Synthesis and Training

Run the following commands from the repository root. Synthesis can run on a CPU; model training requires PyTorch and the relevant model dependencies to be installed separately.

```bash
python -m pip install -r synthesis_pipeline/requirements.txt
python synthesis_pipeline/run_pipeline.py --list-steps
python synthesis_pipeline/run_pipeline.py --config synthesis_pipeline/config.yaml
```

Configuration paths are resolved relative to the **directory containing the YAML file**. Default inputs are located in `synthesis_pipeline/data/`, and outputs are written to `synthesis_pipeline/output/`. The same configuration and random seed reproduce the generation process. Use empty output directories for a fresh run to avoid mixing new samples with previous outputs.

The default splitting utility stratifies by category and background source to create train, val, test/private, and test/public splits in a 70/10/10/10 ratio. **This is a configurable utility split, not a reconstruction of the paper's original 7:3 split indices.** Both background sources contribute to training and validation. `test_private` contains synthetic images and must not be treated as Fab-test; `test_public` must not be directly equated with the paper's Open-test. The paper's sample-level split lists and complete experimental environment have not been supplied as part of this maintenance update.

Install and train YOLO using the Ultralytics version bundled in this repository:

```bash
python -m pip install -e ./yolo
python yolo/trains.py --data synthesis_pipeline/output/synthetic_yolo/data.yaml --model yolo11n-seg.pt --device 0 --imgsz 384 --epochs 50 --batch 64
```

`--model` requires a segmentation model or segmentation architecture. Supply the corresponding checkpoint to use another YOLO version. The bundled version is recorded in `yolo/ultralytics/__init__.py`; these commands do not rely on automatically installing the latest release.

Swin-UNet export uses binary defect labels by default: background is 0 and defect is 1, matching a model with 2 output channels. For the optional eight-category semantic segmentation setting, set `swinunet.binary: false` and use `--num_classes 9`. This constitutes a different experimental setting.

```bash
python -m pip install -r unet/requirements.txt
cd unet
python train.py --root_path ../synthesis_pipeline/output/swin_unet_data --dataset LithoDefect --output_dir ./model_out --num_classes 2 --img_size 224 --batch_size 32 --max_epochs 50
python test.py --root_path ../synthesis_pipeline/output/swin_unet_data --dataset LithoDefect --output_dir ./model_out --num_classes 2 --split_name test_public
```

Fetch the pretrained Swin-T checkpoint through LFS before training. PNG data are read from the existing train/val directories, and validation uses no random augmentation. Legacy NPZ inputs continue to support an explicit `--list_dir`. See the [SAM tool documentation](sam/SAM-Tool/README.md) for installation, annotation, and prompting instructions.

## Mask Evaluation

```bash
python compute.py --gt-dir data/masks --pred-dir results/pred_masks --output-dir results --suffix _mask
```

Prediction filenames follow `<GT filename stem><suffix>.png`. Pass `--suffix ""` when predictions use the same filenames as the ground-truth masks. Missing predictions, empty datasets, corrupted files, or mismatched dimensions raise errors. When dimension alignment is required, explicitly enable `--resize`, which uses nearest-neighbor interpolation.

The output file, `metrics_results.txt`, records the sample count, actual filenames, per-image metrics, and aggregate metrics. The default `--average image` computes a macro-average over images. If filenames begin with `<Category>_`, use `--average category` to average within each category first and then give each category equal weight. These aggregation rules are distinct and should not be mixed. Pixel values greater than 0 are treated as foreground; metrics with a zero denominator are assigned 0.

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

Please report code or dataset issues through [GitHub Issues](https://github.com/Hqa77880011/Benchmark/issues).
