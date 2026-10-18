# Input Data for the Bundled Swin-UNet Loader

For LithoDefectV1, follow the root [dataset preparation and training instructions](../../README.md) or the [Swin-UNet guide](../README.md). The released archive must be prepared before it can be passed to the loader.

## PNG Inputs

`Synapse_dataset` also loads the benchmark's exported PNG layout:

```text
<root>/
├── train/img/*.png
├── train/mask/*.png
├── val/img/*.png
├── val/mask/*.png
├── test/img/*.png
└── test/mask/*.png
```

Image and mask filenames match. Images are converted to RGB. Binary masks store background=0 and defect=1; optional category masks store labels 0–8. Set `--num_classes` to 2 or 9 respectively. Stored PNG labels are not thresholded or renumbered by the loader.

Training reads `train`, validation reads `val`, and inference reads the folder selected by `--split_name`. Existing splits are not divided again. Only training uses random geometric augmentation; validation resizes deterministically. Raw image values remain on the 0–255 scale.

The separate synthesis exporter uses `test_private` and `test_public` instead of `test`. Pass the appropriate actual folder name rather than assuming either one is the manuscript's Fab-test or Open-test set.

## Legacy NPZ Inputs

Pass `--root_path <npz_root> --dataset datasets --list_dir <list_folder>`. The list folder contains `train.txt`, `val.txt`, and the file corresponding to the inference `--split_name`.

Each nonempty list line identifies an NPZ path relative to the data root, with or without `.npz`; a comma-separated suffix is ignored. Each NPZ contains either `image` and `label`, or `data` and `seg`. The benchmark training transforms expect an image shaped `(H, W, C)` and a label shaped `(H, W)`.

Historical upstream Synapse medical-volume instructions are retained in [UPSTREAM.md](../UPSTREAM.md) for reference. The current benchmark loader does not provide an H5 medical-volume evaluation path.
