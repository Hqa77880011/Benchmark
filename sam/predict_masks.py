"""Fixed image-only contrast prompts and deterministic SAM foreground masks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, median_filter


def contrast_points(image, window=15, quantile=.995, minimum_contrast=12., maximum=32):
    """Find local contrast peaks without labels, masks or category information."""
    gray = np.asarray(Image.fromarray(image).convert("L"), dtype=float)
    contrast = np.abs(gray - median_filter(gray, size=window, mode="reflect"))
    threshold = max(minimum_contrast, float(np.quantile(contrast, quantile)))
    components, count = label(contrast >= threshold)
    peaks = []
    for component in range(1, count + 1):
        y, x = np.where(components == component)
        index = int(np.argmax(contrast[y, x]))
        peaks.append((float(contrast[y[index], x[index]]), int(y[index]), int(x[index])))
    return [(x, y) for _, y, x in sorted(peaks, key=lambda p: (-p[0], p[1], p[2]))[:maximum]]


def select_candidate(masks, scores, point, max_area=.15, min_score=.8):
    """Select only by SAM confidence, prompt inclusion and fixed area limits."""
    masks, scores = np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)
    if masks.ndim != 3 or len(masks) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("SAM must return finite scores and candidate masks")
    x, y = point
    areas = masks.mean(axis=(1, 2))
    valid = [i for i in range(len(scores))
             if masks[i, y, x] and 0 < areas[i] <= max_area and scores[i] >= min_score]
    return max(valid, key=lambda i: (scores[i], -areas[i], -i)) if valid else None


def predict_mask(image, predictor, window=15, quantile=.995, minimum_contrast=12.,
                 maximum=32, max_area=.15, min_score=.8, total_area=.35):
    """Produce one binary foreground mask, including an explicit empty prediction."""
    points = contrast_points(image, window, quantile, minimum_contrast, maximum)
    foreground = np.zeros(image.shape[:2], dtype=bool)
    decisions = []
    if points:
        predictor.set_image(image)
    for point in points:
        masks, scores, _ = predictor.predict(point_coords=np.array([point], dtype=float),
                                             point_labels=np.ones(1, dtype=int),
                                             multimask_output=True)
        index = select_candidate(masks, scores, point, max_area, min_score)
        accepted = False
        if index is not None:
            proposed = foreground | masks[index].astype(bool)
            if proposed.mean() <= total_area:
                foreground, accepted = proposed, True
        decisions.append(dict(point=list(point), selected=index, accepted=accepted,
                              score=None if index is None else float(scores[index])))
    return foreground, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--quantile", type=float, default=.995)
    parser.add_argument("--minimum-contrast", type=float, default=12.)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--max-area", type=float, default=.15)
    parser.add_argument("--min-score", type=float, default=.8)
    parser.add_argument("--total-area", type=float, default=.35)
    args = parser.parse_args()
    if (args.window < 3 or args.window % 2 != 1 or not 0 < args.quantile <= 1
            or args.minimum_contrast <= 0 or args.max_prompts < 1
            or not 0 < args.max_area <= args.total_area <= 1 or not 0 <= args.min_score <= 1):
        parser.error("Use an odd window >=3 and valid contrast, prompt, score and area limits")
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError("Use a nonempty cohort with unique image stems")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty prediction directory")
    from segment_anything import SamPredictor, sam_model_registry
    predictor = SamPredictor(sam_model_registry[args.model_type](checkpoint=str(args.checkpoint)).to(args.device))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for file in files:
        with Image.open(file) as source:
            image = np.asarray(source.convert("RGB"))
        foreground, decisions = predict_mask(image, predictor, args.window, args.quantile,
                                             args.minimum_contrast, args.max_prompts,
                                             args.max_area, args.min_score, args.total_area)
        Image.fromarray(foreground.astype(np.uint8) * 255).save(args.output / (file.stem + ".png"))
        records.append(dict(image=str(file.resolve()), prediction=file.stem + ".png",
                            foreground_pixels=int(foreground.sum()), decisions=decisions))
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "protocol.json").write_text(json.dumps(dict(
        protocol="image_contrast_sam_v1", settings=settings, cohort=records
    ), indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
