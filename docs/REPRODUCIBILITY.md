# Reproducibility Notes and Maintenance Scope

Status: the manuscript's major revision has been completed, and the manuscript remains under review. This page documents the relationship between the current code and the paper's experiments without changing historical experimental records.

## Relationship Between the Paper and the Code

- The revised manuscript reports 2,518 Train images, 1,082 Open-test images, 300 Re-synth-test images, and 92 Fab-test images. SAM uses a subset of 129 Open-test images and the other two complete test sets.
- The default synthesis configuration provides a runnable workflow from source material to generated data. The sample count depends on the amount of source material and the number of augmentations; it does not guarantee regeneration of exactly 3,600 images.
- The synthesis utility's 70/10/10/10 split does not use the same indices as the paper's 7:3 split. `private/public` denotes synthetic background sources; `test_private` is not the real Fab-test set.
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
python -m pip install -r synthesis_pipeline/requirements.txt
python -m pip install torch tensorboardX pytest
python -m pytest tests/test_workflow.py -q
```

The regression checks cover synthesis, splitting, and export to both formats from a different working directory; reproducibility with the same seed; binary masks; rejection of missing predictions; hand-calculated metrics; invalid splits; independent training and validation samples; deterministic validation; PNG loading; and erroneous foreground predictions. They use small, artificially constructed images and are not paper experiments.

Validation from the preceding code update: 13 regression checks passed, including one CPU training epoch with a small convolutional network, 20 singleton batches, independent validation, and checkpoint writing. This check validates the training entry point and repaired execution paths. It does not train the Swin-UNet backbone or recompute the paper's metrics.
