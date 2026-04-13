"""
Convert a YOLO-format dataset to COCO JSON annotations.

Usage (from D-FINE/ folder):
    python convert_yolo_to_coco.py
    python convert_yolo_to_coco.py --splits train val test
    python convert_yolo_to_coco.py --yolo-dataset data_final --output-dir custom/dataset/annotations

Coordinate conversion
---------------------
YOLO stores normalised bounding boxes as:
    class_id  x_center  y_center  width  height   (all in [0, 1])

COCO expects absolute pixel coordinates:
    [x_min, y_min, width_px, height_px]

Derivation::

    x_min     = (x_center - width  / 2) * image_width
    y_min     = (y_center - height / 2) * image_height
    width_px  = width  * image_width
    height_px = height * image_height
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import dfine_root, get_class_names, image_size, load_dataset_yaml


# ── Coordinate conversion ─────────────────────────────────────────────────────

def yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    w_norm: float,
    h_norm: float,
    img_w: int,
    img_h: int,
) -> list[float]:
    """Convert a normalised YOLO bounding box to an absolute COCO bbox.

    Args:
        x_center: Normalised horizontal centre of the box.
        y_center: Normalised vertical centre of the box.
        w_norm:   Normalised box width.
        h_norm:   Normalised box height.
        img_w:    Image width in pixels.
        img_h:    Image height in pixels.

    Returns:
        ``[x_min, y_min, width_px, height_px]`` in pixels.
    """
    width_px  = w_norm  * img_w
    height_px = h_norm  * img_h
    x_min     = (x_center - w_norm  / 2) * img_w
    y_min     = (y_center - h_norm  / 2) * img_h
    return [round(x_min, 2), round(y_min, 2), round(width_px, 2), round(height_px, 2)]


# ── Per-split conversion ──────────────────────────────────────────────────────

def convert_split(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    split: str,
    verbose: bool = True,
) -> dict:
    """Build a COCO JSON dict for one dataset split.

    Args:
        images_dir: Root images folder (the split sub-folder is appended).
        labels_dir: Root labels folder (the split sub-folder is appended).
        class_names: Ordered list of class name strings.
        split:      One of ``"train"``, ``"val"``, ``"test"``.
        verbose:    Whether to print progress.

    Returns:
        A dict with keys ``images``, ``annotations``, ``categories``.

    Raises:
        FileNotFoundError: If the images directory for the split is missing.
        ValueError:        If no images are found in the directory.
        ValueError:        If a label line does not have exactly 5 fields.
    """
    split_images = images_dir / split
    split_labels = labels_dir / split

    if not split_images.exists():
        raise FileNotFoundError(
            f"Images directory not found for split '{split}': {split_images}"
        )

    # ── Categories ────────────────────────────────────────────────────────────
    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    # ── Collect image files ───────────────────────────────────────────────────
    image_files: list[Path] = sorted(
        list(split_images.glob("*.jpg")) + list(split_images.glob("*.png"))
    )

    if not image_files:
        raise ValueError(
            f"No .jpg or .png images found in {split_images}. "
            "Verify the dataset path is correct."
        )

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 0
    missing_labels = 0
    skipped_images = 0
    skip_examples: list[str] = []
    img_id = 0

    for img_path in image_files:
        try:
            w, h = image_size(img_path)
        except Exception as exc:
            skipped_images += 1
            if verbose and len(skip_examples) < 5:
                skip_examples.append(f"{img_path.name} ({exc})")
            continue

        coco_images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        label_path = split_labels / f"{img_path.stem}.txt"

        if not label_path.exists():
            # Treat as an unannotated image — valid in COCO (zero annotations).
            missing_labels += 1
            img_id += 1
            continue

        with open(label_path) as fh:
            lines = fh.read().strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue  # skip blank lines

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 fields per YOLO line, got {len(parts)} "
                    f"in {label_path}: '{line}'"
                )

            class_id            = int(parts[0])
            x_c, y_c, bw, bh   = map(float, parts[1:])

            bbox = yolo_to_coco_bbox(x_c, y_c, bw, bh, w, h)
            area = bbox[2] * bbox[3]

            coco_annotations.append(
                {
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": class_id,
                    "bbox":        bbox,
                    "area":        round(area, 2),
                    "iscrowd":     0,
                }
            )
            ann_id += 1

        img_id += 1

    if verbose:
        skip_msg = f" | {skipped_images} invalid/skipped" if skipped_images else ""
        print(
            f"  [{split:5s}] {len(coco_images):4d} images | "
            f"{len(coco_annotations):5d} annotations | "
            f"{missing_labels} images without labels{skip_msg}"
        )
        if skip_examples:
            print("    (examples: " + "; ".join(skip_examples) + ")")

    if not coco_images:
        raise ValueError(
            f"No valid images in {split_images} after reading {len(image_files)} paths. "
            "Files may be corrupt, zero-byte, or Git LFS pointer stubs — replace them with real JPEG/PNG."
        )

    return {
        "images":      coco_images,
        "annotations": coco_annotations,
        "categories":  categories,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a YOLO-format dataset to COCO JSON annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--yolo-dataset",
        type=Path,
        default=dfine_root() / "data_final",
        help="Root of the YOLO dataset (must contain images/, labels/, dataset.yaml).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "custom" / "dataset" / "annotations",
        help="Directory where COCO JSON files will be written.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to convert.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    yolo_root  = args.yolo_dataset.resolve()
    output_dir = args.output_dir.resolve()

    # ── Validate inputs ───────────────────────────────────────────────────────
    dataset_yaml = load_dataset_yaml(yolo_root / "dataset.yaml")
    class_names  = get_class_names(dataset_yaml)

    print(f"YOLO root  : {yolo_root}")
    print(f"Output dir : {output_dir}")
    print(f"Classes    : {class_names}")
    print(f"Splits     : {args.splits}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = yolo_root / "images"
    labels_dir = yolo_root / "labels"

    # ── Convert each split ────────────────────────────────────────────────────
    for split in args.splits:
        coco_data = convert_split(images_dir, labels_dir, class_names, split)

        out_path = output_dir / f"instances_{split}.json"
        with open(out_path, "w") as fh:
            json.dump(coco_data, fh)  # no indent to keep file sizes small

        print(f"  → Saved {out_path.name}")

    print("\nDone. All COCO JSON files written.")


if __name__ == "__main__":
    main()
