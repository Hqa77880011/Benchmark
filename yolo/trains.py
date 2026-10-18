"""Train a YOLO segmentation checkpoint on an explicitly supplied dataset."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Segmentation model YAML/checkpoint")
    parser.add_argument("--device", default="0", help="CPU or CUDA device(s)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="lithodefect")
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        parser.error(f"Dataset YAML not found: {data_path}")
    from ultralytics import YOLO

    model = YOLO(args.model, task="segment")
    if model.task != "segment":
        raise ValueError(f"Expected a segmentation model, got task={model.task}")
    model.train(
        data=str(data_path), imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, workers=args.workers, device=args.device,
        seed=args.seed, project=args.project, name=args.name,
        optimizer="SGD", lr0=0.01, amp=True,
        degrees=10, translate=0.1, scale=0.3, shear=5,
        flipud=0.3, fliplr=0.5, hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        mosaic=1.0, mixup=0.2, copy_paste=0.2,
    )


if __name__ == "__main__":
    main()
