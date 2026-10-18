# Swin-UNet for LithoDefectV1

Network architecture is retained from the bundled Swin-Unet implementation. The original project notes and citation are in [UPSTREAM.md](UPSTREAM.md). This folder's train/test entrypoints support exported PNG pairs as well as the legacy NPZ layout.

## Prepare

From this folder:

```bash
python -m pip install -r requirements.txt
git lfs pull --include="unet/pretrained_ckpt/*.pth,unet/best_model.pth"
```

For new data, run the synthesis exporter from the repository root. Binary export uses 0=background, 1=defect with `--num_classes 2`. Optional category export requires `--num_classes 9`.

## Train and infer

```bash
python train.py --dataset LithoDefect --root_path ../synthesis_pipeline/output/swin_unet_data --output_dir model_out --num_classes 2 --batch_size 32 --max_epochs 50
python test.py --dataset LithoDefect --root_path ../synthesis_pipeline/output/swin_unet_data --output_dir model_out --num_classes 2 --split_name test_public
```

PNG input is `<root>/<split>/img/*.png` with matching `<root>/<split>/mask/*.png`. Training reads `train`, validation reads `val`, and validation transforms are deterministic. Inputs retain the legacy raw 0–255 pixel scale; this maintenance does not introduce new normalization or change the architecture. PNG predictions are discrete class-index masks, suitable for the root `compute.py` (use `--suffix ""`).

For legacy NPZ inputs, pass `--dataset datasets --root_path <npz_root> --list_dir <list_folder>`. Lists are read exactly as supplied. A NPZ contains `image,label` or `data,seg`. `--n_class` remains an alias of `--num_classes`.

The old validation loader used training data. The fixed code and the historical paper results must be distinguished; see [reproducibility notes](../docs/REPRODUCIBILITY.md). Existing checkpoints were not retrained during this update. Set the output channel count to match the checkpoint being loaded.
