"""
Geometric transformation functions for defect augmentation.

Implements the randomized spatial transformations described in Section III-B:
  - Random horizontal/vertical flips
  - Random rotation (0~360°)
  - Random scaling (configurable range)
  - Random geometric distortion (stretch, perspective, barrel)

All transforms are applied synchronously to both image and mask.
"""

import cv2
import numpy as np
import random


def apply_random_flips(image, mask, enable_hflip=True, enable_vflip=True):
    """Apply random horizontal and/or vertical flips to image and mask."""
    if enable_hflip and random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if enable_vflip and random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


def apply_random_rotation(image, mask):
    """Apply random rotation (0~360°) to image and mask."""
    angle = random.uniform(0, 360)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def apply_random_scale(image, mask, scale_range=(0.7, 1.3)):
    """Apply random scaling. Mask uses INTER_NEAREST to preserve class values."""
    lo, hi = scale_range
    scale = random.uniform(lo, hi)
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)
    return image, mask


def apply_random_distortion(image, mask,
                            distortion_types=("stretch", "perspective", "barrel"),
                            stretch_range=(0.7, 1.4),
                            perspective_offset=0.1,
                            barrel_k1_range=(-0.3, 0.3),
                            barrel_k2_range=(-0.1, 0.1)):
    """Apply a randomly selected geometric distortion type."""
    h, w = image.shape[:2]
    dist_type = random.choice(distortion_types)

    if dist_type == "stretch":
        lo, hi = stretch_range
        sx, sy = random.uniform(lo, hi), random.uniform(lo, hi)
        image = cv2.resize(image, None, fx=sx, fy=sy,
                           interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, None, fx=sx, fy=sy,
                          interpolation=cv2.INTER_NEAREST)

    elif dist_type == "perspective":
        offset = int(min(w, h) * perspective_offset)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, M, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    elif dist_type == "barrel":
        k1 = random.uniform(*barrel_k1_range)
        k2 = random.uniform(*barrel_k2_range)
        K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
        d = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        image = cv2.undistort(image, K, -d)
        mask = cv2.undistort(mask, K, -d)

    return image, mask


def apply_transforms(image, mask,
                     enable_hflip=True, enable_vflip=True,
                     enable_rotation=True, scale_range=(0.7, 1.3),
                     enable_distortion=True, **distortion_kwargs):
    """
    Full transformation pipeline: flip → rotate → scale → distort.
    Each call uses fresh random parameters.
    """
    image, mask = apply_random_flips(image, mask, enable_hflip, enable_vflip)
    if enable_rotation:
        image, mask = apply_random_rotation(image, mask)
    image, mask = apply_random_scale(image, mask, scale_range)
    if enable_distortion:
        image, mask = apply_random_distortion(image, mask, **distortion_kwargs)
    return image, mask
