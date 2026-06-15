"""
Expand synthetic_defects dataset by 3x and append new data to all downstream datasets.

- private: 50 → 150 (add 100 per defect)
- public:  30 → 90  (add 60 per defect)
- New data uses aug indices starting from the existing max + 1 to avoid filename conflicts
- After generation, merges into synthetic_defects/, then rebuilds split/yolo/swin_unet

Usage:
  1. Edit the CONFIG dict below with your paths
  2. Run: python tools/expand_data.py
  3. Then re-run: python run_pipeline.py --step split
                    python run_pipeline.py --step yolo
                    python run_pipeline.py --step swinunet
"""

import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# Configuration — Edit these paths to match your data layout
# ============================================================
CONFIG = {
    "defect_img_dir": "data/defects/images",      # Seed defect images
    "defect_mask_dir": "data/defects/masks",       # Seed defect masks
    "bg_dir": "data/backgrounds",                  # Background images (256x256)
    "existing_data": "output/synthetic_defects",   # Existing synthetic data

    # Temporary output for new data
    "new_data_dir": "output/synthetic_defects_new",

    # Background groups & number of ADDITIONAL samples per seed defect
    "bg_groups": {
        "private": {"prefixes": ["luhu_", "chip_"], "add_aug": 100},  # 50 → 150
        "public":  {"prefixes": ["shen_"],           "add_aug": 60},   # 30 → 90
    },

    # Source defect → target category mapping
    "category_mapping": {
        "arching":  ["ArchDeform", "Spindle"],
        "particle": ["ParticleContam", "Block"],
        "peeling":  ["Flake", "Droplet"],
        "residue":  ["ResidueLeft", "Needle"],
    },

    # Synthesis parameters (same as config.yaml)
    "enable_feather": True,
    "feather_kernel_size": 5,
    "enable_hflip": True,
    "enable_vflip": True,
    "enable_rotation": True,
    "scale_range": (0.7, 1.3),
    "enable_distortion": True,
    "distortion_types": ["stretch", "perspective", "barrel"],
    "stretch_range": (0.7, 1.4),
    "perspective_offset": 0.1,
    "barrel_k1_range": (-0.3, 0.3),
    "barrel_k2_range": (-0.1, 0.1),
    "avoid_center": True,
    "center_region_ratio": 0.5,
    "avoid_center_max_attempts": 30,
}

# ============================================================
# Transform and Composition Functions (same as lithodefect package)
# ============================================================

def apply_random_flips(image, mask):
    if CONFIG["enable_hflip"] and random.random() > 0.5:
        image = cv2.flip(image, 1); mask = cv2.flip(mask, 1)
    if CONFIG["enable_vflip"] and random.random() > 0.5:
        image = cv2.flip(image, 0); mask = cv2.flip(mask, 0)
    return image, mask


def apply_random_rotation(image, mask):
    angle = random.uniform(0, 360)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def apply_random_scale(image, mask):
    lo, hi = CONFIG["scale_range"]
    s = random.uniform(lo, hi)
    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    return image, mask


def apply_random_distortion(image, mask):
    h, w = image.shape[:2]
    dist_type = random.choice(CONFIG["distortion_types"])
    if dist_type == "stretch":
        lo, hi = CONFIG["stretch_range"]
        sx, sy = random.uniform(lo, hi), random.uniform(lo, hi)
        image = cv2.resize(image, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, None, fx=sx, fy=sy, interpolation=cv2.INTER_NEAREST)
    elif dist_type == "perspective":
        offset = int(min(w, h) * CONFIG["perspective_offset"])
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, M, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif dist_type == "barrel":
        k1 = random.uniform(*CONFIG["barrel_k1_range"])
        k2 = random.uniform(*CONFIG["barrel_k2_range"])
        K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        d = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        image = cv2.undistort(image, K, -d)
        mask = cv2.undistort(mask, K, -d)
    return image, mask


def apply_transforms(image, mask):
    image, mask = apply_random_flips(image, mask)
    if CONFIG["enable_rotation"]:
        image, mask = apply_random_rotation(image, mask)
    image, mask = apply_random_scale(image, mask)
    if CONFIG["enable_distortion"]:
        image, mask = apply_random_distortion(image, mask)
    return image, mask


def feather_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.GaussianBlur(eroded, (kernel_size, kernel_size), 0)


def get_paste_position(bg_w, bg_h, fg_w, fg_h):
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)
    if not CONFIG["avoid_center"]:
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        return x, y
    ratio = CONFIG["center_region_ratio"]
    cw, ch = int(bg_w * ratio), int(bg_h * ratio)
    cx0, cy0 = (bg_w - cw) // 2, (bg_h - ch) // 2
    for attempt in range(CONFIG["avoid_center_max_attempts"]):
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        in_center = (x >= cx0 and x + fg_w <= cx0 + cw
                     and y >= cy0 and y + fg_h <= cy0 + ch)
        if not in_center or attempt == CONFIG["avoid_center_max_attempts"] - 1:
            break
    return x, y


def composite(foreground, mask, background):
    bg = background.copy()
    bg_h, bg_w = bg.shape[:2]
    if foreground.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    fg_h, fg_w = foreground.shape[:2]
    if fg_h >= bg_h or fg_w >= bg_w:
        s = min(bg_h / fg_h, bg_w / fg_w) * 0.95
        new_w, new_h = int(fg_w * s), int(fg_h * s)
        foreground = cv2.resize(foreground, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)
        fg_h, fg_w = new_h, new_w
    if fg_h >= bg_h:
        fg_h = bg_h - 1
        foreground = foreground[:fg_h, :]
        mask = mask[:fg_h, :]
    if fg_w >= bg_w:
        fg_w = bg_w - 1
        foreground = foreground[:, :fg_w]
        mask = mask[:, :fg_w]
    x, y = get_paste_position(bg_w, bg_h, fg_w, fg_h)
    full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    mk = feather_mask(mask, CONFIG["feather_kernel_size"]) if CONFIG["enable_feather"] else mask
    alpha = np.expand_dims(mk.astype(np.float32) / 255.0, axis=-1)
    roi = bg[y:y + fg_h, x:x + fg_w]
    blended = (roi.astype(np.float32) * (1 - alpha)
               + foreground.astype(np.float32) * alpha)
    bg[y:y + fg_h, x:x + fg_w] = blended.astype(np.uint8)
    full_mask[y:y + fg_h, x:x + fg_w] = mk
    return bg, full_mask


# ============================================================
# Stage 1: Generate new data
# ============================================================

def find_existing_max_aug(existing_dir, bg_group):
    """Find the max aug index for each (defect_type, base_name) in existing data."""
    max_augs = defaultdict(int)
    img_root = os.path.join(existing_dir, bg_group, "img")
    if not os.path.isdir(img_root):
        return max_augs
    for cat in os.listdir(img_root):
        for fname in os.listdir(os.path.join(img_root, cat)):
            if not fname.endswith('.png'):
                continue
            # Parse: particle_000_augXX_bgYYY.png
            parts = fname.split('_aug')
            if len(parts) < 2:
                continue
            prefix = parts[0]  # e.g. "particle_000"
            aug_part = parts[1]  # e.g. "0_bgluhu_000058.png"
            aug_str = aug_part.split('_')[0]
            try:
                aug_num = int(aug_str)
                max_augs[prefix] = max(max_augs[prefix], aug_num)
            except ValueError:
                continue
    return max_augs


def load_backgrounds():
    bg_dir = CONFIG["bg_dir"]
    groups = {name: [] for name in CONFIG["bg_groups"]}
    for fname in os.listdir(bg_dir):
        if not fname.lower().endswith('.png'):
            continue
        fpath = os.path.join(bg_dir, fname)
        for group_name, cfg in CONFIG["bg_groups"].items():
            if any(fname.startswith(p) for p in cfg["prefixes"]):
                img = cv2.imread(fpath)
                if img is not None:
                    groups[group_name].append((fpath, img))
                break
    for name, imgs in groups.items():
        print(f"  {name}: {len(imgs)} backgrounds")
    return groups


def generate_new_data():
    """Generate additional augmented data into new_data_dir."""
    mapping = CONFIG["category_mapping"]
    new_root = CONFIG["new_data_dir"]
    existing = CONFIG["existing_data"]

    print("Loading backgrounds...")
    bg_groups = load_backgrounds()

    defect_types = [d for d in os.listdir(CONFIG["defect_img_dir"])
                    if os.path.isdir(os.path.join(CONFIG["defect_img_dir"], d))
                    and d not in ("good", "fail") and d in mapping]

    for group_name, cfg in CONFIG["bg_groups"].items():
        add_aug = cfg["add_aug"]
        bg_list = bg_groups.get(group_name, [])
        if not bg_list:
            print(f"  WARNING: No backgrounds for '{group_name}', skipping")
            continue

        max_augs = find_existing_max_aug(existing, group_name)
        print(f"\n{'='*50}")
        print(f"Generating {group_name} expansion (+{add_aug} per defect)...")

        total_new = 0
        for defect_type in defect_types:
            target_categories = mapping[defect_type]
            img_dir = os.path.join(CONFIG["defect_img_dir"], defect_type)
            mask_dir = os.path.join(CONFIG["defect_mask_dir"], defect_type)
            img_files = sorted([f for f in os.listdir(img_dir)
                                if f.lower().endswith('.png')])

            for img_name in tqdm(img_files, desc=f"  {group_name}/{defect_type}"):
                base = os.path.splitext(img_name)[0]
                prefix = f"{defect_type}_{base}"
                start_idx = max_augs.get(prefix, -1) + 1

                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, f"{base}_mask.png")
                defect_img = cv2.imread(img_path)
                defect_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if defect_img is None or defect_mask is None:
                    continue

                for aug_idx in range(start_idx, start_idx + add_aug):
                    fg = defect_img.copy()
                    mk = defect_mask.copy()
                    fg, mk = apply_transforms(fg, mk)

                    bg_path, bg_img = random.choice(bg_list)
                    bg_base = os.path.splitext(os.path.basename(bg_path))[0]
                    result_img, result_mask = composite(fg, mk, bg_img.copy())

                    target_cat = target_categories[aug_idx % len(target_categories)]
                    out_img_dir = os.path.join(new_root, group_name, "img", target_cat)
                    out_mask_dir = os.path.join(new_root, group_name, "mask", target_cat)
                    os.makedirs(out_img_dir, exist_ok=True)
                    os.makedirs(out_mask_dir, exist_ok=True)

                    out_name = f"{defect_type}_{base}_aug{aug_idx}_bg{bg_base}.png"
                    cv2.imwrite(os.path.join(out_img_dir, out_name), result_img)
                    cv2.imwrite(os.path.join(out_mask_dir, out_name), result_mask)
                    total_new += 1

        print(f"  {group_name} new: {total_new} images")

    return new_root


# ============================================================
# Stage 2: Merge new data into synthetic_defects/
# ============================================================

def merge_into_existing():
    """Copy files from new_data_dir into existing_data category directories."""
    new_root = CONFIG["new_data_dir"]
    existing = CONFIG["existing_data"]

    for bg_group in ["private", "public"]:
        for sub in ["img", "mask"]:
            src_root = os.path.join(new_root, bg_group, sub)
            dst_root = os.path.join(existing, bg_group, sub)
            if not os.path.isdir(src_root):
                continue
            for cat in os.listdir(src_root):
                src_dir = os.path.join(src_root, cat)
                dst_dir = os.path.join(dst_root, cat)
                if not os.path.isdir(src_dir):
                    continue
                os.makedirs(dst_dir, exist_ok=True)
                for fname in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, fname)
                    dst_path = os.path.join(dst_dir, fname)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)


# ============================================================
# Main
# ============================================================

def main():
    # Clean temp
    new_root = CONFIG["new_data_dir"]
    if os.path.exists(new_root):
        shutil.rmtree(new_root)

    # Stage 1: Generate new data
    print("=" * 55)
    print("Stage 1: Generating expansion data...")
    generate_new_data()

    # Stage 2: Merge into synthetic_defects/
    print(f"\n{'='*55}")
    print("Stage 2: Merging into synthetic_defects/ ...")
    merge_into_existing()

    # Count merged results
    existing = CONFIG["existing_data"]
    for bg in ["private", "public"]:
        img_root = os.path.join(existing, bg, "img")
        total = sum(len(os.listdir(os.path.join(img_root, c)))
                    for c in os.listdir(img_root)
                    if os.path.isdir(os.path.join(img_root, c)))
        print(f"  {bg}: {total} images (after merge)")

    # Clean temp
    shutil.rmtree(new_root)
    print(f"\nMerge complete! Re-run split/yolo/swinunet steps to update downstream datasets.")
    print(f"  python run_pipeline.py --step split")
    print(f"  python run_pipeline.py --step yolo")
    print(f"  python run_pipeline.py --step swinunet")


if __name__ == "__main__":
    main()
