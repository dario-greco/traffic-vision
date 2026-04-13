"""
Prepare the COCO-format dataset directory that D-FINE training expects.

This script:
1. Converts YOLO labels → COCO JSON (via :mod:`convert_yolo_to_coco`).
2. Creates the ``custom/dataset/images/{train,val,test}`` directories and
   populates them with **symlinks** pointing at the original YOLO images.
   Symlinks avoid duplicating large image files on disk.

Expected output layout (relative to D-FINE/)::

    custom/
    └── dataset/
        ├── annotations/
        │   ├── instances_train.json
        │   ├── instances_val.json
        │   └── instances_test.json
        └── images/
            ├── train/  → absolute path into data_final/images/train/  (symlinks per image)
            ├── val/    → …/val/
            └── test/   → …/test/

Usage (from D-FINE/ folder):
    python prepare_dfine_dataset.py
    python prepare_dfine_dataset.py --yolo-dataset data_final --dataset-dir custom/dataset
    python prepare_dfine_dataset.py --splits train val            # skip test
    python prepare_dfine_dataset.py --copy                        # copy instead of symlink
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from convert_yolo_to_coco import convert_split
from utils import dfine_root, get_class_names, load_dataset_yaml


# ── Image linking helpers ─────────────────────────────────────────────────────

def _link_images(src_dir: Path, dst_dir: Path, *, copy: bool) -> int:
    """Populate *dst_dir* with symlinks (or copies) of images in *src_dir*.

    Existing targets are skipped (idempotent).

    Args:
        src_dir: Folder containing source images.
        dst_dir: Destination folder (will be created if needed).
        copy:    If ``True``, copy files instead of symlinking.

    Returns:
        Number of files processed.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img in sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png")):
        dst = dst_dir / img.name
        if dst.exists() or dst.is_symlink():
            continue  # already in place

        if copy:
            shutil.copy2(img, dst)
        else:
            # Resolve to an absolute path so the symlink works regardless of cwd
            dst.symlink_to(img.resolve())
        count += 1

    return count


# ── Main preparation logic ────────────────────────────────────────────────────

def prepare(
    yolo_root: Path,
    dataset_dir: Path,
    splits: list[str],
    copy_images: bool,
    verbose: bool = True,
) -> None:
    """Run the full dataset preparation pipeline.

    Args:
        yolo_root:    Root of the YOLO dataset.
        dataset_dir:  Destination D-FINE dataset directory.
        splits:       List of split names, e.g. ``["train", "val", "test"]``.
        copy_images:  Whether to copy images instead of symlinking them.
        verbose:      Print progress messages.
    """
    # ── Load class names ──────────────────────────────────────────────────────
    yaml_data   = load_dataset_yaml(yolo_root / "dataset.yaml")
    class_names = get_class_names(yaml_data)

    ann_dir = dataset_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if verbose:
            print(f"\n── {split} ──")

        # 1. Convert annotations
        coco_data = convert_split(
            images_dir  = yolo_root / "images",
            labels_dir  = yolo_root / "labels",
            class_names = class_names,
            split       = split,
            verbose     = verbose,
        )

        import json
        ann_path = ann_dir / f"instances_{split}.json"
        with open(ann_path, "w") as fh:
            json.dump(coco_data, fh)
        if verbose:
            print(f"  Annotations → {ann_path.relative_to(dataset_dir.parent)}")

        # 2. Link / copy images
        src_images = yolo_root / "images" / split
        dst_images = dataset_dir / "images" / split
        n = _link_images(src_images, dst_images, copy=copy_images)
        verb = "Copied" if copy_images else "Symlinked"
        if verbose:
            print(f"  {verb} {n} new image(s) → {dst_images.relative_to(dataset_dir.parent)}")

    if verbose:
        print(
            f"\nDataset ready at: {dataset_dir}\n"
            "  annotations/ contains COCO JSON files.\n"
            "  images/      contains per-split image links.\n"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a COCO-layout dataset directory for D-FINE training. "
            "Converts YOLO labels to COCO JSON and symlinks images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--yolo-dataset",
        type=Path,
        default=dfine_root() / "data_final",
        help="Root of the source YOLO dataset (images/, labels/, dataset.yaml).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).parent / "custom" / "dataset",
        help="Destination directory for the prepared D-FINE dataset.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to prepare.",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (use on filesystems that don't support symlinks).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    yolo_root   = args.yolo_dataset.resolve()
    dataset_dir = args.dataset_dir.resolve()

    print(f"YOLO source : {yolo_root}")
    print(f"D-FINE dest : {dataset_dir}")
    print(f"Splits      : {args.splits}")
    print(f"Image mode  : {'copy' if args.copy else 'symlink'}")

    prepare(
        yolo_root   = yolo_root,
        dataset_dir = dataset_dir,
        splits      = args.splits,
        copy_images = args.copy,
    )


if __name__ == "__main__":
    main()
