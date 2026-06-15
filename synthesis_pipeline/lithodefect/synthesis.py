"""
Core synthesis engine for LithoDefectV1 dataset generation.

This module implements the full "Cut-Paste-Learn" synthesis pipeline
described in Section III-B of the paper:

  Stage 1: Defect Extraction — load seed defects and backgrounds
  Stage 2: Geometric Augmentation — random flips, rotation, scale, distortion
  Stage 3: Image Composition — feathering + alpha blending + quality control

Usage:
    from lithodefect.synthesis import Synthesizer

    syn = Synthesizer(config)
    syn.run()
"""

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict

from .transforms import apply_transforms
from .composite import composite


class Synthesizer:
    """
    Synthesizer for generating LithoDefectV1-style defect segmentation data.

    Takes defect "seed" images + public background images → produces
    synthetic images with pixel-level segmentation masks.
    """

    def __init__(self, config):
        """
        Args:
            config: dict with the following keys:
                - defect_img_dir: Path to seed defect images
                - defect_mask_dir: Path to seed defect masks
                - bg_dir: Path to background images
                - output_root: Output directory
                - category_mapping: {source_defect_type: [target_category1, ...]}
                - bg_groups: {group_name: {prefixes: [...], num_aug: N}}
                - synthesis_params: {enable_feather, feather_kernel_size, ...}
        """
        self.config = config
        self._bg_cache = {}

    def load_backgrounds(self):
        """Load and group background images by prefix."""
        bg_dir = self.config["bg_dir"]
        groups = {name: [] for name in self.config["bg_groups"]}

        for fname in os.listdir(bg_dir):
            if not fname.lower().endswith('.png'):
                continue
            fpath = os.path.join(bg_dir, fname)
            for group_name, cfg in self.config["bg_groups"].items():
                if any(fname.startswith(p) for p in cfg["prefixes"]):
                    img = cv2.imread(fpath)
                    if img is not None:
                        groups[group_name].append((fpath, img))
                    break

        for name, imgs in groups.items():
            print(f"  [{name}] {len(imgs)} backgrounds loaded")
        self._bg_cache = groups
        return groups

    def generate_group(self, group_name, bg_list, num_aug):
        """
        Generate synthetic data for one background group.

        Args:
            group_name: 'private' or 'public'
            bg_list: List of (path, image) tuples
            num_aug: Number of augmented samples per seed defect
        """
        params = self.config.get("synthesis_params", {})
        mapping = self.config["category_mapping"]
        output_root = self.config["output_root"]

        defect_img_dir = self.config["defect_img_dir"]
        defect_mask_dir = self.config["defect_mask_dir"]
        skip_dirs = self.config.get("skip_dirs", ["good", "fail"])

        defect_types = [d for d in os.listdir(defect_img_dir)
                        if os.path.isdir(os.path.join(defect_img_dir, d))
                        and d not in skip_dirs and d in mapping]

        total = 0

        for defect_type in defect_types:
            target_categories = mapping[defect_type]
            img_dir = os.path.join(defect_img_dir, defect_type)
            mask_dir = os.path.join(defect_mask_dir, defect_type)

            img_files = sorted([f for f in os.listdir(img_dir)
                                if f.lower().endswith('.png')])

            for img_name in tqdm(img_files, desc=f"  {group_name}/{defect_type}"):
                img_path = os.path.join(img_dir, img_name)
                base = os.path.splitext(img_name)[0]
                mask_path = os.path.join(mask_dir, f"{base}_mask.png")

                defect_img = cv2.imread(img_path)
                defect_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if defect_img is None or defect_mask is None:
                    continue

                for aug_idx in range(num_aug):
                    # Stage 2: Geometric augmentation (each iteration = fresh random params)
                    fg = defect_img.copy()
                    mk = defect_mask.copy()
                    fg, mk = apply_transforms(fg, mk, **params)

                    # Stage 3: Image composition
                    bg_path, bg_img = random.choice(bg_list)
                    bg_base = os.path.splitext(os.path.basename(bg_path))[0]
                    result_img, result_mask = composite(fg, mk, bg_img.copy(), **params)

                    # Round-robin assignment to target categories
                    target_cat = target_categories[aug_idx % len(target_categories)]

                    out_img_dir = os.path.join(output_root, group_name, "img", target_cat)
                    out_mask_dir = os.path.join(output_root, group_name, "mask", target_cat)
                    os.makedirs(out_img_dir, exist_ok=True)
                    os.makedirs(out_mask_dir, exist_ok=True)

                    out_name = f"{defect_type}_{base}_aug{aug_idx}_bg{bg_base}.png"
                    cv2.imwrite(os.path.join(out_img_dir, out_name), result_img)
                    cv2.imwrite(os.path.join(out_mask_dir, out_name), result_mask)
                    total += 1

        return total

    def run(self):
        """Execute the full synthesis pipeline."""
        print("=" * 55)
        print("Stage 1: Defect Extraction — Loading seed defects & backgrounds...")
        bg_groups = self.load_backgrounds()

        grand_total = 0
        for group_name, cfg in self.config["bg_groups"].items():
            num_aug = cfg["num_aug"]
            bg_list = bg_groups.get(group_name, [])
            if not bg_list:
                print(f"  WARNING: No backgrounds for '{group_name}', skipping.")
                continue

            print(f"\nStage 2 & 3: Geometric Augmentation + Image Composition")
            print(f"  Group: {group_name} | {num_aug}x per defect | {len(bg_list)} backgrounds")
            count = self.generate_group(group_name, bg_list, num_aug)
            print(f"  → {count} images generated")
            grand_total += count

        print(f"\n{'=' * 55}")
        print(f"Pipeline complete! Total: {grand_total} synthetic images")
        print(f"Output: {self.config['output_root']}")
        print(f"{'=' * 55}")
        return grand_total
