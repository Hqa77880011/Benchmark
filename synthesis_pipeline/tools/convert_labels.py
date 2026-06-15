"""
Mask → YOLO label converter — convert binary/grayscale masks to YOLO training labels.

Supports two output modes:
  - "detect"  : YOLO detection format (bbox), each line: class_id x_center y_center w h
  - "segment" : YOLO segmentation format (polygon), each line: class_id x1 y1 x2 y2 ...

Features:
  - Auto-extracts class prefix from filename (registered in CLASS_MAPPING)
  - Small contour filtering (area threshold)
  - Adaptive thresholding (OTSU) or manual binarization
  - Recursive subdirectory batch processing
  - Empty labels generate empty .txt files

Usage:
  1. Edit the CONFIG dict below with your paths and parameters
  2. Run: python convert_labels.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# User Configuration
# ============================================================
CONFIG = {
    # --- Paths ---
    "mask_dir": "",           # Input mask directory
    "output_dir": "",         # Output label directory

    # --- Mode ---
    "mode": "detect",         # "detect" = YOLO bbox / "segment" = YOLO polygon

    # --- Class Mapping ---
    # Filename format: <class_prefix>_<number>_...png
    # e.g. ArchDeform_001_aug_0_mask.png → prefix "ArchDeform"
    "class_mapping": {
        "ArchDeform": 0,
        "Block": 1,
        "Droplet": 2,
        "Flake": 3,
        "Needle": 4,
        "ParticleContam": 5,
        "ResidueLeft": 6,
        "Spindle": 7,
    },

    # --- Binarization ---
    "use_otsu": True,             # True=OTSU auto threshold, False=use binary_thresh
    "binary_thresh": 1,           # Manual threshold (only when use_otsu=False)

    # --- Contour Filtering ---
    "min_contour_area": 100,      # Minimum contour area to keep

    # --- Segmentation Mode ---
    "approx_epsilon_factor": 0.002,  # approxPolyDP epsilon factor
}


# ============================================================
# Utility Functions
# ============================================================

def extract_class_id(filename):
    """Extract class prefix from filename and look up class_id."""
    base = os.path.splitext(filename)[0]
    # Strip optional _mask suffix
    if base.endswith("_mask"):
        base = base[:-5]
    # Take the part before the first underscore as class prefix
    prefix = base.split("_")[0]
    return CONFIG["class_mapping"].get(prefix)


def binarize_mask(mask):
    """Binarize mask using OTSU or manual threshold."""
    if CONFIG["use_otsu"]:
        _, binary = cv2.threshold(mask, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(mask, CONFIG["binary_thresh"], 255,
                                  cv2.THRESH_BINARY)
    return binary


# ============================================================
# YOLO Detection Format (bbox)
# ============================================================

def mask_to_yolo_detect(mask, class_id):
    """
    Convert mask to YOLO detection format.
    Returns: "class_id x_center y_center w h" string, or None.
    """
    binary = binarize_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Merge all contours into one bounding box
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    height, width = mask.shape

    # YOLO normalized bbox: [x_center, y_center, width, height] ∈ [0,1]
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w_norm = w / width
    h_norm = h / height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


# ============================================================
# YOLO Segmentation Format (polygon)
# ============================================================

def mask_to_yolo_segment(mask, class_id):
    """
    Convert mask to YOLO segmentation format (polygon points).
    Returns: multi-line string, each line "class_id x1 y1 x2 y2 ...", or None.
    """
    binary = binarize_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = binary.shape
    lines = []

    for contour in contours:
        if cv2.contourArea(contour) < CONFIG["min_contour_area"]:
            continue

        # Simplify contour (reduce number of points)
        epsilon = CONFIG["approx_epsilon_factor"] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.squeeze()

        if points.ndim != 2 or len(points) < 3:
            continue  # Need at least 3 points for a polygon

        # Normalize coordinates
        normalized = points / [w, h]
        line = f"{class_id} " + " ".join(
            f"{p:.6f}" for p in normalized.flatten())
        lines.append(line)

    return "\n".join(lines) if lines else None


# ============================================================
# Batch Processing
# ============================================================

def process_directory(mask_dir, output_dir):
    """Walk through mask directory (including subdirectories) and convert to YOLO labels."""
    mask_files = []
    for root, _, files in os.walk(mask_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                mask_files.append(os.path.join(root, f))

    converter = (mask_to_yolo_detect if CONFIG["mode"] == "detect"
                 else mask_to_yolo_segment)

    empty_count = 0
    for mask_path in tqdm(mask_files, desc="Converting"):
        # Build output path (preserve subdirectory structure)
        rel_path = os.path.relpath(mask_path, mask_dir)
        label_path = os.path.join(output_dir,
                                  os.path.splitext(rel_path)[0] + ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        filename = os.path.basename(mask_path)
        class_id = extract_class_id(filename)
        if class_id is None:
            print(f"  WARNING: Unknown class, skipping {filename}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: Cannot read {mask_path}")
            continue

        result = converter(mask, class_id)
        with open(label_path, "w") as f:
            if result:
                f.write(result)
            else:
                empty_count += 1
                # Empty file means no object in this image

    print(f"Done! {len(mask_files)} masks → {output_dir}")
    print(f"  Empty labels: {empty_count}")


def main():
    mask_dir = CONFIG["mask_dir"]
    output_dir = CONFIG["output_dir"]
    mode = CONFIG["mode"]

    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        return
    if mode not in ("detect", "segment"):
        print(f"ERROR: mode must be 'detect' or 'segment', got: {mode}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Mode: {'YOLO Detection (bbox)' if mode == 'detect' else 'YOLO Segmentation (polygon)'}")
    process_directory(mask_dir, output_dir)


if __name__ == "__main__":
    main()
