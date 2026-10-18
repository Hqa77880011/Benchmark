"""Download an official SAM checkpoint without running model inference."""

import argparse
from pathlib import Path
import shutil
from urllib.request import urlopen

SOURCE = "https://github.com/facebookresearch/segment-anything#model-checkpoints"
CHECKPOINTS = {
    "vit_b": ("sam_vit_b_01ec64.pth", 375042383),
    "vit_l": ("sam_vit_l_0b3195.pth", 1249524607),
    "vit_h": ("sam_vit_h_4b8939.pth", 2564550879),
}


def download_checkpoint(model_type, output):
    """Stream the official release and reject incomplete local/downloaded files."""
    name, expected = CHECKPOINTS[model_type]
    output = Path(output)
    if output.exists():
        if output.stat().st_size != expected:
            raise ValueError(f"Unexpected checkpoint size; move this file aside: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    url = "https://dl.fbaipublicfiles.com/segment_anything/" + name
    with urlopen(url, timeout=120) as response, partial.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    if partial.stat().st_size != expected:
        raise OSError(f"Incomplete checkpoint: {partial}; expected {expected} bytes")
    partial.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=list(CHECKPOINTS), default="vit_b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print official URL without downloading")
    args = parser.parse_args()
    name, expected = CHECKPOINTS[args.model_type]
    output = args.output or Path("checkpoints") / f"sam_{args.model_type}.pth"
    if args.plan:
        print(f"Source: {SOURCE}\nModel: {args.model_type}\n"
              f"URL: https://dl.fbaipublicfiles.com/segment_anything/{name}\n"
              f"Bytes: {expected}\nOutput: {output}")
        return
    print(download_checkpoint(args.model_type, output))


if __name__ == "__main__":
    main()
