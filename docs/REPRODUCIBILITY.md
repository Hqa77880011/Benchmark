# Reproducibility Notes and Maintenance Scope

Status: the manuscript's major revision has been completed, and the manuscript remains under review. This page documents the relationship between the current code and the paper's experiments without changing historical experimental records.

## Relationship Between the Paper and the Code

The current selected reproduction setting uses the saved seed-42 source-grouped utility split for all models. `sam/predict_masks.py` supplies a complete image-only contrast-prompted SAM baseline on every utility test image, with fixed proposal selection and one original-size binary prediction per image. Its `protocol.json` records the cohort and parameters; it never reads ground truth. This protocol replaces the unspecified historical SAM selection rule for new comparisons. Use a newly trained 2-class Swin checkpoint, the chosen YOLO segmentation architecture and the official matching SAM checkpoint acquired by `sam/download_checkpoint.py`. The archived paper table below remains historical; new settings produce their own measured outputs.

- The revised manuscript reports 2,518 Train images, 1,082 Open-test images, 300 Re-synth-test images, and 92 Fab-test images. SAM uses a subset of 129 Open-test images and the other two complete test sets.
- The default synthesis configuration provides a runnable workflow from source material to generated data. The sample count depends on the amount of source material and the number of augmentations; it does not guarantee regeneration of exactly 3,600 images.
- `prepare_inputs.py` obtains the LFS archive and safely extracts paired PNGs. Released-data preparation groups `_aug_N` variants before splitting: 47/8/17 original groups and 2,350/400/850 images in train/val/test. Block has two originals and therefore no validation group. `splits.json` records each original and all image assignments. The target ratio is 70/10/20, with group independence and nonempty train/test prioritized. The separate synthesis utility's 70/10/10/10 split also does not use the same indices as the paper's 7:3 split. `private/public` denotes synthetic background sources; `test_private` is not the real Fab-test set.
- The paper's sample-level split indices, complete per-sample predictions, and full execution environment are currently unavailable in this repository. The README results table is therefore a transcription of the revised manuscript, not the output of this maintenance run.

## Fixes and the Boundary of Historical Results

The original validation loader in `unet/trainer.py` used `db_train` as its input, and validation transforms included random augmentation. The current implementation uses an independent validation set and deterministic preprocessing. When running the old entry point, this defect allows training samples to influence validation loss and the selection of the best checkpoint. The repository alone does not establish whether the paper's training runs used exactly that entry point, nor does it establish that historical models were trained under the corrected protocol.

The paper's reported values and existing checkpoint resources were retained. Models were not retrained, and paper results were not recomputed during this maintenance. To verify how an existing checkpoint was selected, inspect the corresponding training code, training logs, and actual train/val indices.

Additional fixes:

- Restored Swin-UNet text source files that had incorrectly been stored as LFS pointers. Large datasets and checkpoints remain managed with LFS.
- Routed geometric augmentation and image blending settings only to functions that support them, fixing the `TypeError` in the default synthesis entry point.
- Resolved configuration paths relative to the YAML directory and fixed the synthesis random seed and file traversal order.
- Used nearest-neighbor interpolation for labels during geometric transforms. Feathering affects image blending only, and output masks retain discrete labels.
- Included the background group in synthesized filenames to prevent cross-group overwrites during export. Missing masks and unknown categories raise errors.
- Added support for exported PNG split directories in Swin-UNet while retaining legacy NPZ inputs. Existing data splits are not split again.
- Unified the class-count argument while retaining the `--n_class` alias. The YOLO entry point uses segmentation models and configurable paths and devices.
- Required complete prediction pairing during evaluation, preserved the correspondence between actual filenames and metrics, and made dimension alignment opt-in.
- Corrected the legacy Swin-UNet auxiliary Dice calculation so that a foreground prediction on empty ground truth no longer receives a score of 1. The primary evaluation uses IoU, Dice, Precision, and Recall from `compute.py`.

## Resource Management

Existing LFS resources are retained at `data.zip`, `unet/*.pth`, `unet/pretrained_ckpt/*.pth`, and `yolo/best_*`. Text source files are stored as regular Git files. `.gitignore` excludes `__pycache__`, bytecode, training logs, generated datasets, predictions, experiment outputs, editor settings, and temporary files.

Upstream material removed from the working repository includes YOLO's documentation site, examples, tests, and Docker/CI configuration, as well as SAM's web demo and notebooks. These files remain in Git history, and local copies are archived under `archive/upstream_reference/` in the paper project directory. Model implementations, packaging files, original documentation, and licenses are retained.

## Minimal Validation

```bash
python -X utf8 -m pip install -r synthesis_pipeline/requirements.txt -r unet/requirements.txt -e ./yolo pytest
python -m pytest tests/test_workflow.py tests/test_reproduction.py -q
```

The regression checks cover synthesis, splitting, and export to both formats from a different working directory; reproducibility with the same seed; binary masks; rejection of missing predictions; hand-calculated metrics; invalid splits; independent training and validation samples; deterministic validation; PNG loading; and erroneous foreground predictions. They use small, artificially constructed images and are not paper experiments.

## New-User Workflow Verification

The current check used a virtual environment that reused the machine's existing PyTorch runtime. Missing project dependencies were installed into that environment, and pip was upgraded before rechecking the editable installs. The tested CPU runtime was Python 3.13.5, PyTorch 2.7.1, torchvision 0.22.1, NumPy 2.1.3, SciPy 1.15.3, timm 1.0.30, and bundled Ultralytics 8.3.151. This is a current workflow check, not a reconstruction of the original paper environment.

Verified results:

- Downloaded the actual LFS `data.zip`, checked the complete ZIP CRC, and inspected all image headers: 3,600 RGB images and 3,600 grayscale masks, all 360 × 360, with complete same-name pairing across eight categories.
- Prepared a 10-image real-data subset into 7 train, 1 validation, and 2 test images; both model formats and saved split indices were generated. These small subsets validate execution only.
- Ran the actual Swin-UNet backbone for one CPU training epoch using the pretrained encoder, selected a checkpoint on the independent validation image, exported both test masks, and evaluated them with `compute.py`.
- Ran YOLO11n-seg from its architecture YAML for one CPU epoch, saved a trained checkpoint, exported both test masks including empty detections, and evaluated them with `compute.py`.
- Verified SAM dependency installation and the annotation/export command-line parsers. Interactive GUI behavior and full ViT-H embedding/ONNX export were not executed in this check.
- The related regression suite passes 20 checks, including Unicode image paths, saved utility splits, standard YOLO YAML keys, original-size prediction masks, independent validation, and foreground metric boundaries.

The YOLO training run revealed a corrupted `train_yolo12` token throughout the bundled code/configuration. Replacing it with the standard `train` token restored the documented API and dataset schema. Of 71 corrected files, 68 then matched the official 8.3.151 wheel exactly; the remaining local changes were retained. Model architecture code was not replaced wholesale.

A Windows OpenMP conflict was traced to mixing the base Conda MKL NumPy/SciPy builds with the PyTorch wheel. The check used isolated pip numerical wheels instead of suppressing that error. New-user instructions create a fresh virtual environment rather than inheriting the base environment.

The published `unet/best_model.pth` was inspected and has 4 output channels. It therefore requires `--num_classes 4`; the original four-channel label mapping is unavailable. New binary training uses 2 channels and its own newly trained checkpoint.

Tiny smoke-run losses and metrics are not benchmark results. The manuscript's 18 result rows remain unchanged, and original split indices and complete historical predictions are still required for exact paper-result reproduction.
