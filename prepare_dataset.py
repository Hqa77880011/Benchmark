"""Prepare the released img/<category>, mask/<category> PNG archive for training."""

import argparse
import json
from pathlib import Path
import random
import re

from synthesis_pipeline.lithodefect.export_swinunet import export_to_swinunet
from synthesis_pipeline.lithodefect.export_yolo import DEFAULT_CLASS_MAPPING, export_to_yolo
from synthesis_pipeline.lithodefect.split import copy_files


def source_group(path):
    """Keep all named augmentation variants of one original sample together."""
    return re.sub(r"_aug_\d+(?=_|$)", "", Path(path).stem)


def split_source_groups(pairs, train_ratio, val_ratio, rng):
    """Split original identities; two-original categories use train/test only."""
    grouped = {}
    for pair in pairs:
        grouped.setdefault(source_group(pair[0]), []).append(pair)
    keys = sorted(grouped)
    rng.shuffle(keys)
    if len(keys) < 2:
        raise ValueError("At least two independent originals per category are required")
    if len(keys) == 2:
        n_train, n_val = 1, 0
    else:
        n_train = min(len(keys) - 2, max(1, int(len(keys) * train_ratio)))
        n_val = min(len(keys) - n_train - 1, max(1, int(len(keys) * val_ratio)))
    roles = {"train": keys[:n_train], "val": keys[n_train:n_train + n_val],
             "test": keys[n_train + n_val:]}
    return {role: [pair for key in ids for pair in grouped[key]] for role, ids in roles.items()}, roles



def prepare_dataset(source, output, train_ratio=0.7, val_ratio=0.1, seed=42,
                    binary=True):
    """Make source-disjoint utility splits and export both model formats."""
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
                "source": str(source), "group_key": "category plus filename with _aug_N removed",
                "source_groups": {}, "category_counts": {}, "splits": {name: [] for name in ("train", "val", "test")}}
    split_tables = {}
    for category, pairs in grouped.items():
        groups, identities = split_source_groups(pairs, train_ratio, val_ratio, rng)
        manifest["source_groups"][category] = identities
        manifest["category_counts"][category] = {role: len(items) for role, items in groups.items()}
        split_tables[category] = groups
        for name, items in groups.items():
            manifest["splits"][name].extend(Path(image).relative_to(source).as_posix()
                                              for image, _ in items)
    if not all(manifest["splits"].values()):
        raise ValueError("Dataset needs nonempty train/val/test; add independent originals")
    for category, groups in split_tables.items():
        for name, items in groups.items():
            copy_files(items, output / "splits", name, category)
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
