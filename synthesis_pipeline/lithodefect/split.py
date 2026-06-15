"""
Dataset splitting module — stratified train/val/test partition.

Splits synthetic dataset into:
  - train (70%): mixed private + public backgrounds
  - val   (10%): mixed private + public backgrounds
  - test/private (10%): private backgrounds only
  - test/public  (10%): public backgrounds only

The split is stratified by (category, background_source) to ensure
each subgroup is proportionally represented in each split.
"""

import os
import random
import shutil
from collections import defaultdict


def gather_samples(src_root):
    """Collect all samples grouped by (category, background_source)."""
    groups = defaultdict(list)

    for bg_src in ["private", "public"]:
        img_root = os.path.join(src_root, bg_src, "img")
        mask_root = os.path.join(src_root, bg_src, "mask")
        if not os.path.isdir(img_root):
            continue

        for cat in os.listdir(img_root):
            cat_img_dir = os.path.join(img_root, cat)
            cat_mask_dir = os.path.join(mask_root, cat)
            if not os.path.isdir(cat_img_dir):
                continue

            for fname in os.listdir(cat_img_dir):
                if not fname.lower().endswith('.png'):
                    continue
                img_path = os.path.join(cat_img_dir, fname)
                mask_path = os.path.join(cat_mask_dir, fname)
                if os.path.exists(mask_path):
                    groups[(cat, bg_src)].append((img_path, mask_path))

    return groups


def copy_files(split_items, out_root, split_name, cat):
    """Copy a batch of samples to the target split directory."""
    for img_path, mask_path in split_items:
        out_img = os.path.join(out_root, split_name, "img", cat,
                               os.path.basename(img_path))
        out_mask = os.path.join(out_root, split_name, "mask", cat,
                                os.path.basename(mask_path))
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)
        shutil.copy2(img_path, out_img)
        shutil.copy2(mask_path, out_mask)


def split_dataset(src_root, out_root,
                  train_ratio=0.70, val_ratio=0.10,
                  test_private_ratio=0.10, test_public_ratio=0.10,
                  seed=42):
    """
    Perform stratified split of the synthetic dataset.

    Args:
        src_root: Path to synthetic_defects/ (output of synthesis step)
        out_root: Output directory for split data
        train_ratio, val_ratio, test_private_ratio, test_public_ratio:
            Split ratios (must sum to 1.0)
        seed: Random seed for reproducibility

    Returns:
        dict with counts for each split
    """
    assert abs(train_ratio + val_ratio + test_private_ratio + test_public_ratio - 1.0) < 1e-6

    random.seed(seed)

    print("Collecting samples...")
    groups = gather_samples(src_root)
    total_all = sum(len(v) for v in groups.values())
    print(f"  Found {total_all} samples across {len(groups)} (category, bg) groups")

    # Per-category statistics
    by_cat = defaultdict(lambda: {"private": 0, "public": 0})
    for (cat, bg), items in groups.items():
        by_cat[cat][bg] = len(items)

    print("  Class distribution:")
    for cat in sorted(by_cat):
        p = by_cat[cat]["private"]
        u = by_cat[cat]["public"]
        print(f"    {cat}: private={p}, public={u}")

    total_private = sum(len(v) for (c, bg), v in groups.items() if bg == "private")
    total_public = sum(len(v) for (c, bg), v in groups.items() if bg == "public")

    private_test_ratio_group = (test_private_ratio * total_all / total_private
                                if total_private > 0 else 0)
    public_test_ratio_group = (test_public_ratio * total_all / total_public
                               if total_public > 0 else 0)

    train_of_remaining = train_ratio / (train_ratio + val_ratio)
    val_of_remaining = val_ratio / (train_ratio + val_ratio)

    stats = {"train": 0, "val": 0, "test/private": 0, "test/public": 0}

    for (cat, bg_src), items in sorted(groups.items()):
        test_ratio = (private_test_ratio_group if bg_src == "private"
                      else public_test_ratio_group)

        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio)) if n >= 3 else 0
        test_items = items[:n_test]
        remaining = items[n_test:]

        random.shuffle(remaining)
        n_rem = len(remaining)
        n_train = int(n_rem * train_of_remaining)
        train_items = remaining[:n_train]
        val_items = remaining[n_train:]

        test_split = "test/private" if bg_src == "private" else "test/public"

        copy_files(train_items, out_root, "train", cat)
        copy_files(val_items, out_root, "val", cat)
        copy_files(test_items, out_root, test_split, cat)

        stats["train"] += len(train_items)
        stats["val"] += len(val_items)
        stats[test_split] += len(test_items)

    total = sum(stats.values())
    print(f"\n{'=' * 55}")
    print("Split complete!")
    for name in ["train", "val", "test/private", "test/public"]:
        cnt = stats[name]
        pct = cnt / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {name:<14s}: {cnt:5d} images ({pct:5.1f}%) {bar}")
    print(f"  {'─' * 40}")
    print(f"  {'Total':<14s}: {total:5d} images")
    print(f"  Output: {out_root}")
    print(f"{'=' * 55}")

    return stats
