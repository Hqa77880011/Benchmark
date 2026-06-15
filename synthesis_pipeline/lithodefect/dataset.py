"""
PyTorch Dataset class for LithoDefectV1 semantic segmentation.

Supports class-stratified train/val/test splits with automatic
image-mask pairing from the standard directory structure.

Directory structure expected:
    data_root/
    ├── img/<ClassA>/001.png ...
    └── mask/<ClassA>/001.png ...
"""

import os
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class LithoDefectDataset(Dataset):
    """
    PyTorch Dataset for lithography defect segmentation.

    Args:
        data_root: Root directory containing img/ and mask/ subdirectories
        transform: Optional albumentations/torchvision transform
        mode: 'train', 'val', or 'test'
        split_ratios: (train, val, test) proportions (must sum to 1.0)
        seed: Random seed for reproducible splits
        save_split: If True, copy split results to disk
    """

    def __init__(self, data_root, transform=None, mode='train',
                 split_ratios=(0.6, 0.2, 0.2), seed=42, save_split=False):
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        self.split_ratios = split_ratios
        self.seed = seed

        self.image_paths = []
        self.mask_paths = []
        self.class_indices = []
        self.class_names = []

        # Auto-detect img/ and mask/ directories
        subdirs = os.listdir(data_root)
        img_dir = next((d for d in subdirs if 'mask' not in d.lower()), None)
        mask_dir = next((d for d in subdirs if 'mask' in d.lower()), None)
        if img_dir is None or mask_dir is None:
            raise ValueError(f"Cannot find img/ and mask/ in {data_root}")

        img_root = os.path.join(data_root, img_dir)
        mask_root = os.path.join(data_root, mask_dir)

        class_idx = 0
        for class_name in sorted(os.listdir(img_root)):
            class_img_dir = os.path.join(img_root, class_name)
            class_mask_dir = os.path.join(mask_root, class_name)
            if not os.path.isdir(class_img_dir) or not os.path.exists(class_mask_dir):
                continue

            images = sorted(f for f in os.listdir(class_img_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg')))
            masks = set(f for f in os.listdir(class_mask_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg')))

            for img_file in images:
                base, ext = os.path.splitext(img_file)
                mask_candidate = img_file
                if mask_candidate not in masks:
                    mask_candidate = f"{base}_mask{ext}"
                if mask_candidate not in masks:
                    continue

                self.image_paths.append(os.path.join(class_img_dir, img_file))
                self.mask_paths.append(os.path.join(class_mask_dir, mask_candidate))
                self.class_indices.append(class_idx)
                self.class_names.append(class_name)
            class_idx += 1

        if len(self.image_paths) == 0:
            raise ValueError("No image-mask pairs found. Check directory structure.")

        self.indices = self._stratified_split()

        if save_split:
            self._save_splits()

    def _stratified_split(self):
        """Class-stratified train/val/test split."""
        unique_classes = np.unique(self.class_indices)
        train_idx, val_idx, test_idx = [], [], []

        for cls in unique_classes:
            cls_indices = np.where(np.array(self.class_indices) == cls)[0]
            train, temp = train_test_split(
                cls_indices,
                test_size=self.split_ratios[1] + self.split_ratios[2],
                random_state=self.seed)
            val_ratio = self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2])
            val, test = train_test_split(temp, test_size=val_ratio,
                                         random_state=self.seed)
            train_idx.extend(train)
            val_idx.extend(val)
            test_idx.extend(test)

        mode_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        if self.mode not in mode_map:
            raise ValueError(f"mode must be 'train'/'val'/'test', got: {self.mode}")
        return mode_map[self.mode]

    def _save_splits(self):
        """Save train/val/test splits to disk."""
        output_dir = os.path.join(self.data_root, 'split_data')
        unique_classes = np.unique(self.class_indices)
        all_train, all_val, all_test = [], [], []
        for cls in unique_classes:
            cls_indices = np.where(np.array(self.class_indices) == cls)[0]
            train, temp = train_test_split(
                cls_indices,
                test_size=self.split_ratios[1] + self.split_ratios[2],
                random_state=self.seed)
            val_ratio = self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2])
            val, test = train_test_split(temp, test_size=val_ratio,
                                         random_state=self.seed)
            all_train.extend(train)
            all_val.extend(val)
            all_test.extend(test)

        for split_name, indices in [('train', all_train),
                                     ('val', all_val),
                                     ('test', all_test)]:
            for idx in indices:
                class_name = self.class_names[idx]
                img_dest = os.path.join(output_dir, split_name, 'images', class_name)
                mask_dest = os.path.join(output_dir, split_name, 'masks', class_name)
                os.makedirs(img_dest, exist_ok=True)
                os.makedirs(mask_dest, exist_ok=True)
                shutil.copy2(self.image_paths[idx],
                             os.path.join(img_dest, os.path.basename(self.image_paths[idx])))
                shutil.copy2(self.mask_paths[idx],
                             os.path.join(mask_dest, os.path.basename(self.mask_paths[idx])))
        print(f"Split data saved to: {output_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = Image.open(self.image_paths[actual_idx]).convert('RGB')
        mask = Image.open(self.mask_paths[actual_idx]).convert('L')

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
        train_ds = LithoDefectDataset(data_root, mode='train', save_split=True)
        val_ds = LithoDefectDataset(data_root, mode='val')
        test_ds = LithoDefectDataset(data_root, mode='test')
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    else:
        print("Usage: python dataset.py <data_root>")
