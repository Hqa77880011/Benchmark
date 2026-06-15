"""
Image composition functions — the core "paste" step of the synthesis pipeline.

Implements the Gaussian feathering + Alpha Blending technique described in
Section III-B.3 of the paper:

    1. Feather the binary mask: erode → Gaussian blur
    2. Alpha blend: I_syn = α ⊙ I_defect + (1-α) ⊙ I_background
    3. Place defect avoiding image center to simulate real spatial distribution
"""

import cv2
import numpy as np
import random


def feather_mask(mask, kernel_size=5):
    """
    Apply feathering to reduce hard edges.
    Step 1: Erode to shrink the mask slightly.
    Step 2: Gaussian blur to create soft transition.
    Corresponds to Equation (2) in the paper: α = M'_d * G_σ
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.GaussianBlur(eroded, (kernel_size, kernel_size), 0)


def get_paste_position(bg_w, bg_h, fg_w, fg_h,
                       avoid_center=True,
                       center_region_ratio=0.5,
                       max_attempts=30):
    """
    Determine paste position, optionally avoiding the image center.
    This simulates real-world defect spatial distribution where defects
    are rarely centered.
    """
    max_x = max(0, bg_w - fg_w)
    max_y = max(0, bg_h - fg_h)

    if not avoid_center:
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        return x, y

    cw, ch = int(bg_w * center_region_ratio), int(bg_h * center_region_ratio)
    cx0, cy0 = (bg_w - cw) // 2, (bg_h - ch) // 2

    for attempt in range(max_attempts):
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        in_center = (x >= cx0 and x + fg_w <= cx0 + cw
                     and y >= cy0 and y + fg_h <= cy0 + ch)
        if not in_center or attempt == max_attempts - 1:
            break
    return x, y


def composite(foreground, mask, background,
              feather_kernel_size=5, enable_feather=True,
              avoid_center=True, center_region_ratio=0.5):
    """
    Paste a foreground defect onto a background using alpha blending.

    This implements Equation (3) from the paper:
        I_syn = α ⊙ I'_d + (1-α) ⊙ I_b(x,y)

    Args:
        foreground: Defect image (H, W, 3) uint8
        mask: Binary mask (H, W) uint8, 255=defect
        background: Background image (H, W, 3) uint8 (typically 256x256)
        feather_kernel_size: Gaussian kernel size for edge feathering
        enable_feather: Whether to apply feathering
        avoid_center: Avoid placing defect at image center
        center_region_ratio: Center avoidance region ratio

    Returns:
        (synthesized_image, full_size_mask) tuple
    """
    bg = background.copy()
    bg_h, bg_w = bg.shape[:2]

    # Ensure foreground and mask have same dimensions
    if foreground.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    fg_h, fg_w = foreground.shape[:2]

    # If foreground >= background, scale down to fit (leave 5% margin)
    if fg_h >= bg_h or fg_w >= bg_w:
        s = min(bg_h / fg_h, bg_w / fg_w) * 0.95
        new_w, new_h = int(fg_w * s), int(fg_h * s)
        foreground = cv2.resize(foreground, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)
        fg_h, fg_w = new_h, new_w

    # Safety: ensure foreground is strictly smaller than background
    if fg_h >= bg_h:
        fg_h = bg_h - 1
        foreground = foreground[:fg_h, :]
        mask = mask[:fg_h, :]
    if fg_w >= bg_w:
        fg_w = bg_w - 1
        foreground = foreground[:, :fg_w]
        mask = mask[:, :fg_w]

    x, y = get_paste_position(bg_w, bg_h, fg_w, fg_h,
                              avoid_center, center_region_ratio)
    full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    mk = feather_mask(mask, feather_kernel_size) if enable_feather else mask

    # Alpha blending: bg * (1 - α) + fg * α
    alpha = np.expand_dims(mk.astype(np.float32) / 255.0, axis=-1)
    roi = bg[y:y + fg_h, x:x + fg_w]
    blended = (roi.astype(np.float32) * (1 - alpha)
               + foreground.astype(np.float32) * alpha)
    bg[y:y + fg_h, x:x + fg_w] = blended.astype(np.uint8)
    full_mask[y:y + fg_h, x:x + fg_w] = mk

    return bg, full_mask
