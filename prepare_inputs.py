"""Obtain the released LFS image-mask archive and export grouped model inputs."""

import argparse
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import zipfile


def extract_pairs(archive, destination):
    """Copy only regular img/mask PNGs under fixed category subdirectories."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use an empty extraction directory: {destination}")
    with zipfile.ZipFile(archive) as bundle:
        selected, names = [], set()
        for member in bundle.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if (len(parts) != 3 or parts[0] not in {"img", "mask"}
                    or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in {".", ".."} for p in parts)
                    or not parts[2].lower().endswith(".png")
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unexpected archive member: {member.filename}")
            name = "/".join(parts)
            if name in names:
                raise ValueError(f"Duplicate archive member: {name}")
            names.add(name)
            selected.append((member, parts))
        images = {n[4:] for n in names if n.startswith("img/")}
        masks = {n[5:] for n in names if n.startswith("mask/")}
        if not images or images != masks:
            raise ValueError("The release must contain exactly paired image and mask PNGs")
        destination.mkdir(parents=True, exist_ok=True)
        for member, parts in selected:
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data.zip"))
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Require an already downloaded archive")
    parser.add_argument("--plan", action="store_true", help="Print acquisition/preparation steps without writing")
    args = parser.parse_args()
    if args.plan:
        print(f"Acquire: git lfs pull --include=data.zip\nArchive: {args.archive}\n"
              f"Extract paired PNGs: {args.source}\n"
              f"Export source-grouped train/val/test: {args.output}; seed={args.seed}\n"
              "SAM weights: python sam/download_checkpoint.py --model-type vit_b")
        return
    for directory in (args.source, args.output):
        if directory.exists() and any(directory.iterdir()):
            parser.error(f"Use an empty directory: {directory}")
    pointer = (args.archive.is_file()
               and args.archive.stat().st_size < 1024
               and args.archive.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"))
    if not args.archive.is_file() or pointer:
        if args.offline:
            parser.error("--offline requires the actual released data.zip")
        root = Path(__file__).resolve().parent
        if args.archive.resolve() != (root / "data.zip").resolve():
            parser.error("Automatic LFS acquisition requires the repository's data.zip; supply custom archives locally")
        subprocess.run(["git", "lfs", "pull", "--include=data.zip"], cwd=root, check=True)
    if not zipfile.is_zipfile(args.archive):
        parser.error("Expected a complete ZIP archive, not an LFS pointer or partial file")
    extract_pairs(args.archive, args.source)
    from prepare_dataset import prepare_dataset
    prepare_dataset(args.source, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
