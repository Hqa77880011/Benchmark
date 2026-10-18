# Reproducibility protocol

The executable protocol uses `prepare_inputs.py` and its saved seed-42 source-grouped split for every model. `_aug_N` variants of an original image remain in one role. `splits.json` records the assignments. Preparation exports YOLO instance labels and binary Swin-UNet masks; each inference entry writes one binary mask per test image.

## Inputs and model resources

- `data.zip` is the released synthetic image/mask archive managed by Git LFS. `prepare_inputs.py` downloads it when necessary and extracts only paired category PNGs. It requires empty extraction/export directories.
- The synthesis source paths and seed are set in `synthesis_pipeline/config.yaml`, relative to the YAML location. The generator's train/val/private-background/public-background roles are distinct from acquired industrial-image roles.
- YOLO receives the prepared `yolo/data.yaml`; `--model` accepts a segmentation architecture or compatible checkpoint.
- Swin-UNet receives the exported PNG role directories, `--num_classes 2`, and the configuration under `unet/configs/`. Encoder initialization is available with `git lfs pull --include="unet/pretrained_ckpt/*.pth"`.
- `unet/best_model.pth` is an archived four-output-channel checkpoint. Its original four-channel label mapping is unavailable; use a newly trained two-channel checkpoint for the current binary protocol.
- Archived YOLO/Swin resources remain accessible through Git LFS. Their presence alone does not establish their training split or validation selection.
- `sam/download_checkpoint.py` obtains the matching upstream ViT-B/L/H release; `sam/predict_masks.py` generates image-only local-contrast prompts. The selected checkpoint and all prompt/cohort settings are recorded with predictions.

## Evaluation

`compute.py` requires complete image-to-prediction pairing and reports binary IoU, Dice, precision and recall using the declared averaging rule. All compared methods should use the same test cohort and saved split. Prediction resizing is opt-in; inference outputs retain original image dimensions.

The original manuscript split indices and complete per-sample predictions are not distributed. The current utility split and image-prompted SAM procedure define new executable comparisons; archived manuscript values are not supplied as current repository results. Real fab-image evaluation requires separately acquired and annotated SEM inputs.

## Validation scope

This README/code-structure update used source syntax checks, configuration parsing and CLI-definition checks. It did not run training, full benchmark evaluation or CI. Existing regression checks remain under `tests/`; upstream source, packaging files, licenses and citations are retained.
