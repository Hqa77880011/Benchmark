"""OpenCV image I/O that also works with Unicode paths on Windows."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Decode an image from bytes without OpenCV's path encoding conversion."""
    content = np.fromfile(path, dtype=np.uint8)
    if not content.size:
        raise OSError(f"Empty image file: {path}")
    image = cv2.imdecode(content, flags)
    if image is None:
        raise OSError(f"Could not decode image: {path}")
    return image


def write_image(path, image):
    """Encode and write an image using the destination file extension."""
    success, encoded = cv2.imencode(Path(path).suffix, image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    encoded.tofile(path)
