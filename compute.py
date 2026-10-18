"""Evaluate complete, paired binary masks with an explicit averaging rule."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def calculate_metrics(pred, true, resize=False):
    """Treat positive pixels as foreground; zero denominators yield zero.

    Two empty masks therefore score zero, preserving the original convention.
    Resizing requires explicit opt-in and always uses nearest interpolation.
    """
    pred_foreground = np.asarray(pred) > 0
    true_foreground = np.asarray(true) > 0
    if pred_foreground.ndim != 2 or true_foreground.ndim != 2:
        raise ValueError("Masks must be two-dimensional arrays.")
    if pred_foreground.shape != true_foreground.shape:
        if not resize:
            raise ValueError(f"Mask shapes differ: {pred_foreground.shape} vs {true_foreground.shape}")
        pred_foreground = np.asarray(Image.fromarray(pred_foreground).resize(
            (true_foreground.shape[1], true_foreground.shape[0]), Image.Resampling.NEAREST))
    tp = np.count_nonzero(true_foreground & pred_foreground)
    fp = np.count_nonzero(~true_foreground & pred_foreground)
    fn = np.count_nonzero(true_foreground & ~pred_foreground)
    return {
        "IoU": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "Dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def evaluate_masks(gt_dir, pred_dir, suffix="_mask", resize=False, average="image"):
    """Evaluate every ground-truth image; missing predictions are errors.

    For category averaging, filenames must start with <Category>_ and each
    category receives equal weight after averaging its image-level metrics.
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    if not gt_dir.is_dir() or not pred_dir.is_dir():
        raise FileNotFoundError("Both ground-truth and prediction directories must exist.")
    files = sorted(p for p in gt_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        raise ValueError(f"No ground-truth masks in {gt_dir}")
    pairs = [(gt, pred_dir / f"{gt.stem}{suffix}.png") for gt in files]
    missing = [str(pred) for _, pred in pairs if not pred.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction(s); first: {missing[0]}")
    records = []
    for gt, pred in pairs:
        with Image.open(gt) as source:
            true = np.array(source) if source.mode in {"P", "L", "I", "1"} else np.array(source.convert("L"))
        with Image.open(pred) as source:
            predicted = np.array(source) if source.mode in {"P", "L", "I", "1"} else np.array(source.convert("L"))
        records.append((gt.stem, calculate_metrics(predicted, true, resize=resize)))
    groups = {}
    for name, metrics in records:
        category = name.split("_", 1)[0] if average == "category" else "all"
        groups.setdefault(category, []).append(metrics)
    summary = {
        key: float(np.mean([np.mean([m[key] for m in items]) for items in groups.values()]))
        for key in records[0][1]
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="_mask")
    parser.add_argument("--resize", action="store_true", help="Explicitly resize predictions to GT dimensions")
    parser.add_argument("--average", choices=["image", "category"], default="image")
    args = parser.parse_args()
    records, summary = evaluate_masks(args.gt_dir, args.pred_dir, args.suffix,
                                      resize=args.resize, average=args.average)
    lines = [f"Samples: {len(records)}", f"Average: {args.average}",
             "Foreground: value > 0; zero denominator: 0", ""]
    for name, metrics in records:
        lines.append(name + ": " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
    lines.append("\nMean: " + ", ".join(f"{k}={v:.6f}" for k, v in summary.items()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "metrics_results.txt"
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[0])
    print(lines[-1])
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
