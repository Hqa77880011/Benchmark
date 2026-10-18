# LithoDefectV1 synthesis pipeline

This folder implements seed loading, synchronized geometric augmentation, alpha composition, stratified splitting and YOLO / Swin-UNet export. The paper's revision status and reported benchmark results are maintained in the [repository README](../README.md).

## Which Data Preparation Path to Use

The released `data.zip` contains final `img/<category>` and `mask/<category>` pairs. To train from that archive, use `python prepare_dataset.py --source data --output output/benchmark` from the repository root after extraction; see the [main README](../README.md). It does not supply the seeds/backgrounds required below.

## Run

From the repository root:

```bash
python -X utf8 -m pip install -r synthesis_pipeline/requirements.txt
python synthesis_pipeline/run_pipeline.py --config synthesis_pipeline/config.yaml
```

The following are alternatives for running individual steps, in order. Do not repeat them against the nonempty outputs of the full-pipeline command above:

```bash
python synthesis_pipeline/run_pipeline.py --step synthesis
python synthesis_pipeline/run_pipeline.py --step split
python synthesis_pipeline/run_pipeline.py --step yolo
python synthesis_pipeline/run_pipeline.py --step swinunet
```

All paths in YAML are relative to that YAML file. The default config reads `synthesis_pipeline/data/` and writes `synthesis_pipeline/output/`. Supply an empty output directory for a fresh run. Set `seed` for reproducibility. The number and resolution of generated pairs follow the supplied seeds, background sizes and `num_aug`; the config does not enforce the published dataset size.

## Inputs

```text
data/
├── defects/images/<source_type>/seed.png
├── defects/masks/<source_type>/seed_mask.png
└── backgrounds/
    ├── chip_001.png       # private source group
    ├── luhu_001.png       # private source group
    └── shen_001.png       # public source group
```

The four example source types are `arching`, `particle`, `peeling`, `residue`; edit `category_mapping` and `class_mapping` for your own material. The mapping assigns target categories round-robin; users must check that their seed morphology supports those category labels. Seed material must be prepared separately; an archive of final benchmark images does not necessarily contain it.

Geometric masks use nearest interpolation. Feathering creates the image blending alpha only; exported ground-truth remains binary.

## Outputs and split meaning

```text
output/
├── synthetic_defects/{private,public}/{img,mask}/<category>/*.png
├── synthetic_defects_split/{train,val,test/private,test/public}/{img,mask}/<category>/*.png
├── synthetic_yolo/
│   ├── data.yaml
│   ├── images/{train,val,test_private,test_public}/
│   └── labels/{train,val,test_private,test_public}/
└── swin_unet_data/{train,val,test_private,test_public}/{img,mask}/*.png
```

The configurable split defaults to 70% train, 10% validation, 10% private-background synthetic test, 10% public-background synthetic test (subject to per-group integer rounding). Both sources contribute to train/val. It does not recreate the paper's 2,518/1,082 index lists. Neither generated test folder is the paper's real industrial Fab-test. Re-synth-test requires a separate run with a different seed and a separate output root.

Swin-UNet export defaults to foreground=1, background=0. Set `swinunet.binary: false` for category labels 1–8; use 9 model output channels for that setting. The model loader reads these existing PNG split folders without re-splitting them.

For Python use, run from this folder or add it to `PYTHONPATH`:

```python
from run_pipeline import load_config, step_synthesis, step_split

config = load_config("config.yaml")
step_synthesis(config)
step_split(config)
```

The optional `lithodefect.dataset.LithoDefectDataset` reads either grouped binary inputs or exported flat PNG pairs. Use `split_ratios=None` for existing split folders. Optional tools in `tools/` have their own input configuration and are not called by the default pipeline.

## Use the Generated Outputs

From the repository root, set the YOLO `--data` argument to `synthesis_pipeline/output/synthetic_yolo/data.yaml`. Set Swin-UNet `--root_path` to `synthesis_pipeline/output/swin_unet_data` and `--split_name` to `test_private` or `test_public`.

For example, after training a binary Swin-UNet model on those outputs:

```bash
python unet/test.py --root_path synthesis_pipeline/output/swin_unet_data --checkpoint output/swin_model/best_model.pth --output_dir output/swin_model --test_save_dir output/synthetic_predictions --num_classes 2 --split_name test_public --device cpu
python compute.py --gt-dir synthesis_pipeline/output/swin_unet_data/test_public/mask --pred-dir output/synthetic_predictions --output-dir output/synthetic_metrics --suffix= --average category
```

These paths refer to freshly synthesized utility data and do not recreate the manuscript's Open-test, Re-synth-test, or Fab-test indices.
