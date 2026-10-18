"""Prepare the released img/<category>, mask/<category> PNG archive for training."""

import argparse
import json
from pathlib import Path
import random

from synthesis_pipeline.lithodefect.export_swinunet import export_to_swinunet
from synthesis_pipeline.lithodefect.export_yolo import DEFAULT_CLASS_MAPPING, export_to_yolo
from synthesis_pipeline.lithodefect.split import copy_files


def prepare_dataset(source, output, train_ratio=0.7, val_ratio=0.1, seed=42,
                    binary=True):
    """Make a class-stratified utility split and export both model formats."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Train/val ratios must be positive and leave a positive test fraction.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    if not (source / "img").is_dir() or not (source / "mask").is_dir():
        raise FileNotFoundError(f"Expected {source}/img/<category> and mask/<category>.")
    grouped = {}
    for directory in sorted((source / "img").iterdir()):
        if not directory.is_dir():
            continue
        if directory.name not in DEFAULT_CLASS_MAPPING:
            raise ValueError(f"Unknown defect category: {directory.name}")
        pairs = []
        for image in sorted(directory.glob("*.png")):
            mask = source / "mask" / directory.name / image.name
            if not mask.is_file():
                raise FileNotFoundError(f"Missing paired mask: {mask}")
            pairs.append((str(image), str(mask)))
        if pairs:
            grouped[directory.name] = pairs
    if not grouped:
        raise ValueError(f"No PNG image-mask pairs in {source}")

    rng = random.Random(seed)
    manifest = {"seed": seed, "train_ratio": train_ratio, "val_ratio": val_ratio,
                "source": str(source), "splits": {name: [] for name in ("train", "val", "test")}}
    for category, pairs in grouped.items():
        rng.shuffle(pairs)
        n_train, n_val = int(len(pairs) * train_ratio), int(len(pairs) * val_ratio)
        if min(n_train, n_val, len(pairs) - n_train - n_val) == 0:
            raise ValueError(f"Not enough samples in {category} for nonempty train/val/test splits.")
        groups = {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val],
                  "test": pairs[n_train + n_val:]}
        for name, items in groups.items():
            copy_files(items, output / "splits", name, category)
            manifest["splits"][name].extend(Path(image).relative_to(source).as_posix()
                                              for image, _ in items)
    split_names = [(name, name) for name in ("train", "val", "test")]
    export_to_yolo(output / "splits", output / "yolo", splits=split_names)
    export_to_swinunet(output / "splits", output / "swin_unet",
                       splits=split_names, binary=binary)
    (output / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {name: len(items) for name, items in manifest["splits"].items()}
    print(f"Prepared utility split: {counts}")
    print("These saved indices are a new split, not the manuscript's original split.")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multiclass", action="store_true", help="Export 9-class Swin-UNet masks instead of binary masks")
    args = parser.parse_args()
    prepare_dataset(args.source, args.output, args.train_ratio, args.val_ratio,
                    args.seed, binary=not args.multiclass)


if __name__ == "__main__":
    main()
