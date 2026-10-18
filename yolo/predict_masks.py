"""Export one binary PNG mask per input image for the root compute.py evaluator."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def save_result_mask(result, output):
    """Merge predicted instances and save a zero mask when nothing is detected."""
    height, width = result.orig_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if result.masks is not None:
        foreground = result.masks.data.cpu().numpy().any(axis=0)
        mask = np.asarray(Image.fromarray(foreground).resize(
            (width, height), Image.Resampling.NEAREST), dtype=np.uint8) * 255
    target = Path(output) / f"{Path(result.path).stem}.png"
    Image.fromarray(mask).save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained segmentation .pt checkpoint")
    parser.add_argument("--source", type=Path, required=True, help="Directory of PNG images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    images = sorted(args.source.glob("*.png"))
    if not images:
        parser.error(f"No PNG images in {args.source}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Use an empty prediction directory: {args.output}")
    from ultralytics import YOLO

    model = YOLO(args.model, task="segment")
    if model.task != "segment":
        parser.error("A segmentation checkpoint is required.")
    args.output.mkdir(parents=True, exist_ok=True)
    for image in images:
        result = model.predict(source=str(image), device=args.device, imgsz=args.imgsz,
                               conf=args.conf, retina_masks=True, verbose=False)[0]
        save_result_mask(result, args.output)
    print(f"Saved {len(images)} binary masks to {args.output}")


if __name__ == "__main__":
    main()
