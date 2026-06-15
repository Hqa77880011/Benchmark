"""
Export synthetic dataset to Swin-UNet format.

Converts the split dataset into Swin-UNet-compatible format:
    swin_unet_data/
    ├── {train,val,test_private,test_public}/
    │   ├── img/    (RGB images, flattened)
    │   └── mask/   (Grayscale, pixel value = class_id, 0 = background)
"""

import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm


# Default class mapping: background=0, defect classes start from 1
DEFAULT_CLASS_MAPPING = {
    "ArchDeform": 1,
    "Block": 2,
    "Droplet": 3,
    "Flake": 4,
    "Needle": 5,
    "ParticleContam": 6,
    "ResidueLeft": 7,
    "Spindle": 8,
}

DEFAULT_SPLITS = [
    ("train", "train"),
    ("val", "val"),
    ("test/private", "test_private"),
    ("test/public", "test_public"),
]


def export_to_swinunet(src_root, out_root, class_mapping=None, splits=None):
    """
    Convert the split dataset to Swin-UNet format.

    Masks are converted from binary (0/255) to class-index format
    where each pixel value equals the integer class ID.

    Args:
        src_root: Path to synthetic_defects_split/
        out_root: Output path for Swin-UNet-formatted data
        class_mapping: dict {category_name: class_id}
        splits: List of (src_subdir, output_name) tuples
    """
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING
    if splits is None:
        splits = DEFAULT_SPLITS

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    print("=" * 50)
    print("Exporting to Swin-UNet format...")
    print(f"  Classes: {len(class_mapping)} (+ background=0)")

    total_all = 0

    for src_subdir, out_name in splits:
        img_src_root = os.path.join(src_root, src_subdir, "img")
        mask_src_root = os.path.join(src_root, src_subdir, "mask")
        out_img_dir = os.path.join(out_root, out_name, "img")
        out_mask_dir = os.path.join(out_root, out_name, "mask")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        samples = []
        for cat in os.listdir(img_src_root):
            cat_img_dir = os.path.join(img_src_root, cat)
            cat_mask_dir = os.path.join(mask_src_root, cat)
            if not os.path.isdir(cat_img_dir):
                continue
            class_id = class_mapping.get(cat)
            if class_id is None:
                print(f"  WARNING: Unknown category '{cat}', skipping.")
                continue

            for fname in os.listdir(cat_img_dir):
                if fname.lower().endswith('.png'):
                    samples.append((os.path.join(cat_img_dir, fname),
                                    os.path.join(cat_mask_dir, fname),
                                    class_id))

        for img_path, mask_path, cls_id in tqdm(samples, desc=f"  {out_name}"):
            # Copy image as-is (already RGB)
            shutil.copy2(img_path, os.path.join(out_img_dir,
                                                os.path.basename(img_path)))

            # Convert mask: binary (0/255) → class-index (0=bg, N=class_id)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            class_mask = np.where(mask > 127, cls_id, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(out_mask_dir, os.path.basename(mask_path)),
                        class_mask)

        print(f"    {out_name}: {len(samples)} images")
        total_all += len(samples)

    print(f"\n{'=' * 50}")
    print(f"Swin-UNet export complete!")
    print(f"  Total images: {total_all}")
    print(f"  Output: {out_root}")
    print(f"  Note: background=0, {class_mapping}")
    print(f"{'=' * 50}")
