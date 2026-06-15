"""
Export synthetic dataset to YOLO segmentation format.

Converts the split dataset structure:
    synthetic_defects_split/<split>/img/<cat>/xxx.png
    synthetic_defects_split/<split>/mask/<cat>/xxx.png

Into YOLO format:
    synthetic_yolo/
    ├── images/{train,val,test_private,test_public}/
    ├── labels/{train,val,test_private,test_public}/
    └── data.yaml
"""

import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm


# Default 8-class mapping for LithoDefectV1
DEFAULT_CLASS_MAPPING = {
    "ArchDeform": 0,
    "Block": 1,
    "Droplet": 2,
    "Flake": 3,
    "Needle": 4,
    "ParticleContam": 5,
    "ResidueLeft": 6,
    "Spindle": 7,
}

DEFAULT_SPLITS = [
    ("train", "train"),
    ("val", "val"),
    ("test/private", "test_private"),
    ("test/public", "test_public"),
]


def mask_to_yolo_polygon(mask, class_id, min_contour_area=50,
                         approx_epsilon_factor=0.002):
    """
    Convert a binary mask to YOLO polygon format string.

    Each line: class_id x1 y1 x2 y2 ... (coordinates normalized to [0,1])

    Args:
        mask: Grayscale mask (H, W) uint8
        class_id: Integer class ID
        min_contour_area: Minimum contour area to keep (filters noise)
        approx_epsilon_factor: Polygon simplification factor

    Returns:
        YOLO polygon string, or None if no valid contours found
    """
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if np.all(binary == 0):
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = binary.shape
    lines = []

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        epsilon = approx_epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.squeeze()

        if points.ndim != 2 or len(points) < 3:
            continue

        normalized = points / [w, h]
        line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized.flatten())
        lines.append(line)

    return "\n".join(lines) if lines else None


def export_to_yolo(src_root, out_root, class_mapping=None, splits=None,
                   min_contour_area=50, approx_epsilon_factor=0.002):
    """
    Convert the split dataset to YOLO segmentation format.

    Args:
        src_root: Path to synthetic_defects_split/
        out_root: Output path for YOLO-formatted data
        class_mapping: dict {category_name: class_id}
        splits: List of (src_subdir, output_name) tuples
        min_contour_area: Minimum contour area filter
        approx_epsilon_factor: Polygon simplification factor
    """
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING
    if splits is None:
        splits = DEFAULT_SPLITS

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    print("=" * 50)
    print("Exporting to YOLO segmentation format...")
    print(f"  Classes: {len(class_mapping)}")
    print(f"  Splits: {[s[1] for s in splits]}")

    grand_total = 0
    grand_empty = 0

    for src_subdir, out_name in splits:
        img_src_root = os.path.join(src_root, src_subdir, "img")
        mask_src_root = os.path.join(src_root, src_subdir, "mask")
        out_img_dir = os.path.join(out_root, "images", out_name)
        out_label_dir = os.path.join(out_root, "labels", out_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        # Collect samples
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

        empty_count = 0
        for img_path, mask_path, class_id in tqdm(samples, desc=f"  {out_name}"):
            # Copy image
            out_img_path = os.path.join(out_img_dir, os.path.basename(img_path))
            if not os.path.exists(out_img_path):
                shutil.copy2(img_path, out_img_path)

            # Generate YOLO label
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                empty_count += 1
                continue

            yolo_text = mask_to_yolo_polygon(mask, class_id,
                                             min_contour_area,
                                             approx_epsilon_factor)
            base = os.path.splitext(os.path.basename(mask_path))[0]
            label_path = os.path.join(out_label_dir, f"{base}.txt")

            with open(label_path, "w") as f:
                if yolo_text:
                    f.write(yolo_text)
                else:
                    empty_count += 1

        print(f"    {out_name}: {len(samples)} images, {empty_count} empty labels")
        grand_total += len(samples)
        grand_empty += empty_count

    # Generate data.yaml
    _generate_yaml(out_root, class_mapping)

    print(f"\n{'=' * 50}")
    print(f"YOLO export complete!")
    print(f"  Total images: {grand_total}")
    print(f"  Empty labels: {grand_empty}")
    print(f"  Output: {out_root}")
    print(f"{'=' * 50}")


def _generate_yaml(out_root, class_mapping):
    """Generate YOLO data.yaml configuration file."""
    names = [""] * len(class_mapping)
    for name, cid in class_mapping.items():
        names[cid] = name

    yaml_content = f"""# YOLO Segmentation Dataset Configuration
# Auto-generated by LithoDefect pipeline
#
# Dataset: LithoDefectV1
# Classes: {len(names)} defect types

path: {out_root}
train: images/train
val: images/val
test: images/test_private
test_public: images/test_public

nc: {len(names)}
names: {names}
"""
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n  data.yaml saved to: {yaml_path}")
