# Swin-UNet for LithoDefectV1

The bundled architecture comes from Swin-Unet; original project notes and citations are in [UPSTREAM.md](UPSTREAM.md). This folder supports the exported PNG splits and legacy NPZ inputs.

## Install and Prepare Data

Run all commands below from the repository root, inside the activated virtual environment described in the [main README](../README.md):

```bash
python -X utf8 -m pip install -r unet/requirements.txt
python -X utf8 -m pip install -r synthesis_pipeline/requirements.txt
git lfs pull --include="data.zip,unet/pretrained_ckpt/*.pth"
python -m zipfile -e data.zip data
python prepare_dataset.py --source data --output output/benchmark --seed 42
```

The release archive contains `img/<category>` and `mask/<category>`, not train/val folders. The preparation command makes a new 70/10/20 utility split, saves `splits.json`, and exports `output/benchmark/swin_unet/{train,val,test}/{img,mask}/*.png`. This is distinct from the paper's original split. Use an empty output directory for a fresh preparation run.

Masks default to background=0 and defect=1, with `--num_classes 2`. An optional `prepare_dataset.py --multiclass` run exports category IDs 1–8 and requires 9 model output channels.

## Train, Predict, and Evaluate

```bash
python unet/train.py --dataset LithoDefect --root_path output/benchmark/swin_unet --output_dir output/swin_model --num_classes 2 --img_size 224 --batch_size 32 --max_epochs 50
python unet/test.py --dataset LithoDefect --root_path output/benchmark/swin_unet --checkpoint output/swin_model/best_model.pth --output_dir output/swin_model --test_save_dir output/swin_predictions --num_classes 2 --split_name test --device cpu
python compute.py --gt-dir output/benchmark/swin_unet/test/mask --pred-dir output/swin_predictions --output-dir output/swin_metrics --suffix= --average category
```

For CPU training, add `--device cpu --num_workers 0` and reduce the batch size. Pretrained checkpoint paths are relative to the model YAML file, so these root-level commands work without changing directories. `--checkpoint` selects the trained model; without it, inference looks for `<output_dir>/best_model.pth`. `--test_save_dir` selects the prediction directory; its default is `<output_dir>/predictions`.

The loader reads existing PNG splits without re-splitting. Training uses random geometric augmentation, validation uses deterministic preprocessing, and both retain the legacy raw 0–255 pixel scale. Predictions are discrete class-index PNGs with the original image dimensions. The auxiliary Dice/HD95 log uses its own empty-mask convention; use `compute.py` for the repository's four foreground mask metrics.

## Existing Checkpoint and Legacy Inputs

The released `unet/best_model.pth` has **4 output channels**. Fetch it with `git lfs pull --include="unet/best_model.pth"` and use `--num_classes 4` when loading it. Its original four-channel label mapping is not supplied. It is separate from the 2-channel binary model trained by the commands above.

For legacy NPZ input, pass `--dataset datasets --root_path <npz_root> --list_dir <list_folder>`. A NPZ contains `image,label` or `data,seg`, and the supplied list files determine the samples. `--n_class` remains an alias of `--num_classes`.

The synthesis workflow instead produces `synthesis_pipeline/output/swin_unet_data/` with `test_private` and `test_public` folders. Set `--root_path` and `--split_name` to that workflow's actual folders rather than mixing its paths with the released-data preparation above.

The old validation loader used training data. See the [reproducibility notes](../docs/REPRODUCIBILITY.md) for the distinction between repaired code and historical results; existing checkpoints have not been retrained as part of repository maintenance.
