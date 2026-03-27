#!/usr/bin/env python3

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _gather_image_label_pairs(images_train: Path, labels_train: Path) -> list[tuple[Path, Path]]:
    if not images_train.exists():
        raise FileNotFoundError(f"Missing images train folder: {images_train}")
    if not labels_train.exists():
        raise FileNotFoundError(f"Missing labels train folder: {labels_train}")

    # YOLO labels are typically .txt with the same stem as the image filename.
    labels_by_stem: dict[str, Path] = {}
    for p in labels_train.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            labels_by_stem[p.stem] = p

    pairs: list[tuple[Path, Path]] = []
    for img in images_train.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_by_stem.get(img.stem)
        if lbl is None:
            continue
        pairs.append((img, lbl))

    return pairs


def _move_one(src: Path, dst_dir: Path, dry_run: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    if not dry_run:
        print(f"Moving {src} -> {dst}")
        shutil.move(str(src), str(dst))
    else:
        print(f"[DRY RUN] Would move {src} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a YOLO test split by moving N samples from train -> test (images + labels)."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../yolo_dataset"),
        help="Dataset root containing images/ and labels/ (default: yolo_dataset)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples to move into test (default: 100)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=946,
        help="Random seed for reproducible sampling (default: 946)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not move any files",
    )
    args = ap.parse_args()

    images_train = args.data_dir / "images" / "train"
    labels_train = args.data_dir / "labels" / "train"
    images_test = args.data_dir / "images" / "test"
    labels_test = args.data_dir / "labels" / "test"

    pairs = _gather_image_label_pairs(images_train, labels_train)
    if len(pairs) < args.n:
        raise ValueError(
            f"Not enough matched image/label pairs in train. Found {len(pairs)} pairs, need {args.n}."
        )

    rng = random.Random(args.seed)
    chosen = rng.sample(pairs, args.n)

    # Move images and labels together, failing fast if any collisions happen.
    for img, lbl in chosen:
        _move_one(img, images_test, args.dry_run)
        _move_one(lbl, labels_test, args.dry_run)

    moved_word = "Would move" if args.dry_run else "Moved"
    print(
        f"{moved_word} {len(chosen)} samples into:\n"
        f"  - {images_test}\n"
        f"  - {labels_test}\n"
        f"Removed them from:\n"
        f"  - {images_train}\n"
        f"  - {labels_train}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# dry run: 
# python3 split_test.py --data-dir "../yolo_dataset" --n 100 --seed 946 --dry-run

# actually move the files:
# python3 split_test.py --data-dir "../yolo_dataset" --n 100 --seed 946