"""Read existing grouped or flat PNG pairs without implicitly splitting them."""

from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LithoDefectDataset(Dataset):
    """Load img/mask pairs; split only when split_ratios is explicitly given.

    Without a transform, return float RGB tensors in [0, 1] and long masks.
    Binary masks use foreground=1. Set binary=False to preserve category IDs.
    A custom transform accepts and returns an (image, mask) pair.
    """

    def __init__(self, data_root, transform=None, mode="train",
                 split_ratios=None, seed=42, save_split=False, binary=True):
        self.data_root = Path(data_root)
        self.transform = transform
        self.binary = binary
        img_root = self.data_root / "img"
        mask_root = self.data_root / "mask"
        if not img_root.is_dir() or not mask_root.is_dir():
            raise ValueError(f"Expected img/ and mask/ directories in {data_root}")
        self.image_paths, self.mask_paths = [], []
        by_class = {}
        for image in sorted(img_root.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                continue
            relative = image.relative_to(img_root)
            mask = mask_root / relative
            if not mask.is_file():
                mask = mask.with_name(f"{mask.stem}_mask{mask.suffix}")
            if not mask.is_file():
                raise FileNotFoundError(f"Missing paired mask: {mask}")
            category = relative.parts[0] if len(relative.parts) > 1 else "all"
            by_class.setdefault(category, []).append(len(self.image_paths))
            self.image_paths.append(image)
            self.mask_paths.append(mask)
        if not self.image_paths:
            raise ValueError(f"No image-mask pairs in {data_root}")
        if split_ratios is None:
            if save_split:
                raise ValueError("save_split requires explicit split_ratios.")
            self.indices = list(range(len(self.image_paths)))
            return
        if len(split_ratios) != 3 or min(split_ratios) < 0 or abs(sum(split_ratios) - 1) > 1e-6:
            raise ValueError("split_ratios must contain three nonnegative values summing to 1.")
        splits = {"train": [], "val": [], "test": []}
        rng = random.Random(seed)
        for indices in by_class.values():
            rng.shuffle(indices)
            n_train = int(len(indices) * split_ratios[0])
            n_val = int(len(indices) * split_ratios[1])
            splits["train"].extend(indices[:n_train])
            splits["val"].extend(indices[n_train:n_train + n_val])
            splits["test"].extend(indices[n_train + n_val:])
        if mode not in splits:
            raise ValueError(f"Unknown mode: {mode}")
        self.indices = splits[mode]
        if save_split:
            for name, indices in splits.items():
                for idx in indices:
                    for source, kind in [(self.image_paths[idx], "img"),
                                         (self.mask_paths[idx], "mask")]:
                        dest = self.data_root / "split_data" / name / kind / source.relative_to(self.data_root / kind)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with Image.open(self.image_paths[actual_idx]) as source:
            image = source.convert("RGB")
        with Image.open(self.mask_paths[actual_idx]) as source:
            mask = np.array(source)
        if self.binary:
            mask = (mask > 0).astype(np.uint8)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.asarray(mask).copy())
        return image, mask.long()
