"""
Advanced image-mask augmentation with Thin-Plate Spline (TPS) deformation.

Augmentation pipeline:
  1. Affine transform (rotation + scaling)
  2. Multi-scale elastic deformation
  3. Optional TPS (Thin-Plate Spline) nonlinear warping

All transforms are applied synchronously to image and mask.
Use this for training-time data augmentation on individual image-mask pairs.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def affine_transform(image, mask, rotation_range=(-25, 25),
                     scale_range=(0.8, 1.2)):
    """Synchronous random rotation + scaling."""
    h, w = mask.shape
    angle = np.random.uniform(*rotation_range)
    scale = np.random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mk = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
    return img, mk


def elastic_transform(image, mask, alpha=70, sigma=14,
                      max_disp_ratio=0.12):
    """
    Multi-scale elastic deformation.
    Superimposes 5 displacement fields at different resolutions
    plus a global directional flow field for natural-looking warping.
    """
    random_state = np.random.RandomState(None)
    h, w = mask.shape
    base = max(h, w) / 256.0
    alpha_scaled = alpha * base

    def random_field(scale, weight):
        noise = random_state.rand(h // scale + 2, w // scale + 2) * 2 - 1
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        return gaussian_filter(noise, sigma) * weight

    # 5-scale displacement field superposition
    weights = {256: 0.55, 128: 0.45, 64: 0.35, 32: 0.25, 16: 0.15}
    dx = sum(random_field(s, w) for s, w in weights.items()) * alpha_scaled
    dy = sum(random_field(s, w) for s, w in weights.items()) * alpha_scaled

    # Global directional flow
    theta = random_state.uniform(-np.pi, np.pi)
    flow_x = np.cos(theta)
    flow_y = np.sin(theta)
    dx += flow_x * random_state.uniform(8, 15) * base
    dy += flow_y * random_state.uniform(8, 15) * base

    # Clip maximum displacement
    max_disp = max(h, w) * max_disp_ratio
    dx = np.clip(dx, -max_disp, max_disp)
    dy = np.clip(dy, -max_disp, max_disp)

    # Strong smoothing to reduce artifacts
    dx = gaussian_filter(dx, sigma=8)
    dy = gaussian_filter(dy, sigma=8)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (y + dy, x + dx)

    # Mask: order=0 (nearest) to preserve class values
    distorted_mask = map_coordinates(mask, indices, order=0, mode='reflect')
    # Image: order=1 (bilinear) per channel
    distorted_image = np.zeros_like(image)
    for c in range(3):
        distorted_image[..., c] = map_coordinates(
            image[..., c], indices, order=1, mode='reflect')

    return distorted_image, distorted_mask


def tps_warp(image, mask, grid_size=4, alpha=0.12, reg=1e-3):
    """
    Thin-Plate Spline nonlinear warping.
    TPS formula: f(x,y) = a0 + a1*x + a2*y + Σ w_i * U(||(x,y)-p_i||)
    where U(r) = r² log(r) is the radial basis function.
    """
    h, w = mask.shape
    base = max(h, w) / 256.0
    nx = ny = grid_size

    # Uniform control point grid
    xs = np.linspace(0, w - 1, nx)
    ys = np.linspace(0, h - 1, ny)
    px, py = np.meshgrid(xs, ys)
    src_pts = np.stack([px.ravel(), py.ravel()], axis=1)
    N = src_pts.shape[0]

    # Random control point displacement
    max_disp = max(h, w) * (alpha * base)
    disp = (np.random.randn(N, 2) * 0.5) * max_disp
    dst_pts = src_pts + disp

    def U(r2):
        eps = 1e-8
        with np.errstate(divide='ignore', invalid='ignore'):
            return r2 * np.log(r2 + eps)

    # Build TPS linear system L * [w; a] = V
    diff = src_pts[:, None, :] - src_pts[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K = U(r2)

    P = np.concatenate([np.ones((N, 1)), src_pts], axis=1)
    top = np.concatenate([K + np.eye(N) * reg, P], axis=1)
    bottom = np.concatenate([P.T, np.zeros((3, 3))], axis=1)
    L = np.concatenate([top, bottom], axis=0)

    Vx = np.concatenate([dst_pts[:, 0], np.zeros(3)])
    Vy = np.concatenate([dst_pts[:, 1], np.zeros(3)])

    try:
        params_x = np.linalg.solve(L, Vx)
        params_y = np.linalg.solve(L, Vy)
    except np.linalg.LinAlgError:
        return image, mask

    w_x, a_x = params_x[:N], params_x[N:]
    w_y, a_y = params_y[:N], params_y[N:]

    # Compute TPS mapping for each pixel
    Xg, Yg = np.meshgrid(np.arange(w), np.arange(h))
    Xg_flat, Yg_flat = Xg.ravel(), Yg.ravel()
    Pq = np.stack([Xg_flat, Yg_flat], axis=1)

    diff_q = Pq[:, None, :] - src_pts[None, :, :]
    r2_q = np.sum(diff_q ** 2, axis=2)
    Uq = U(r2_q)

    fx = a_x[0] + a_x[1] * Xg_flat + a_x[2] * Yg_flat + Uq.dot(w_x)
    fy = a_y[0] + a_y[1] * Xg_flat + a_y[2] * Yg_flat + Uq.dot(w_y)

    map_x = fx.reshape((h, w)).astype(np.float32)
    map_y = fy.reshape((h, w)).astype(np.float32)

    map_x = gaussian_filter(map_x, sigma=1.0)
    map_y = gaussian_filter(map_y, sigma=1.0)

    warped_img = cv2.remap(image, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.remap(mask, map_x, map_y,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REFLECT)
    return warped_img, warped_mask


def augment_single(image, mask, use_tps=True, **kwargs):
    """
    Full augmentation pipeline for a single image-mask pair:
    affine → elastic → optional TPS → binarize mask.

    Args:
        image: RGB image (H, W, 3) uint8
        mask: Grayscale mask (H, W) uint8
        use_tps: Enable TPS warping
        **kwargs: Override default transform parameters

    Returns:
        (augmented_image, augmented_mask) tuple
    """
    img, mk = affine_transform(image, mask, **kwargs)
    img, mk = elastic_transform(img, mk, **kwargs)
    if use_tps:
        img, mk = tps_warp(img, mk, **kwargs)
    mk = (mk > 128).astype(np.uint8) * 255
    return img, mk
