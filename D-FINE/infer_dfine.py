"""
Run D-FINE inference on images and save annotated visualisations.

Loads the fine-tuned (or pre-trained) D-FINE model directly via the official
D-FINE framework and draws detections on the input images using matplotlib.

Works with *any* D-FINE checkpoint that was trained or fine-tuned with the
official D-FINE training code (PyTorch .pth format).

Usage (from D-FINE/ folder):
    # Single image
    python infer_dfine.py --checkpoint output/dfine_m_traffic/best.pth \\
                          --input  ../yolo_dataset/images/test/000000009091.jpg

    # Directory of images
    python infer_dfine.py --checkpoint output/dfine_m_traffic/best.pth \\
                          --input  ../yolo_dataset/images/test/

    # Change confidence threshold
    python infer_dfine.py --checkpoint best.pth --input img.jpg --threshold 0.4

    # Save to a specific output directory
    python infer_dfine.py --checkpoint best.pth --input test/ --out-dir my_preds/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils import (
    add_dfine_src_to_path,
    assert_dfine_cloned,
    dfine_root,
    get_class_names,
    load_dataset_yaml,
)

# Colours for bounding boxes (one per class, cycling if needed)
_BOX_COLOURS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(config_path: Path, checkpoint_path: Path, device: str):
    """Load a D-FINE model from an official checkpoint.

    Uses D-FINE's own ``YAMLConfig`` / ``TASKS`` registry so the model is
    built exactly as during training.

    Args:
        config_path:     Path to the YAML config used during training.
        checkpoint_path: Path to the .pth checkpoint.
        device:          Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(model, postprocessor)`` — the detection model and its
        post-processing callable.
    """
    add_dfine_src_to_path()

    # D-FINE's config / solver machinery
    from core import YAMLConfig  # type: ignore[import]

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))

    if hasattr(cfg, "yaml_cfg"):
        cfg.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = cfg.yaml_cfg.get(
            "eval_spatial_size", [640, 640]
        )

    model       = cfg.model.to(device)
    postprocess = cfg.postprocessor

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    if isinstance(state, dict) and "module" in state:
        state = state["module"]
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, postprocess


# ── Image pre/post-processing ─────────────────────────────────────────────────

def preprocess_image(image_path: Path, device: str) -> tuple:
    """Load and normalise an image for D-FINE inference.

    Args:
        image_path: Path to the input image.
        device:     Target torch device.

    Returns:
        ``(tensor, orig_size)`` where *tensor* is ``[1, 3, H, W]`` (float32,
        normalised to [0, 1]) and *orig_size* is ``(height, width)``.
    """
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)

    transform = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, orig_size


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_detections(
    image_path: Path,
    boxes: list,
    scores: list,
    labels: list,
    class_names: list[str],
    threshold: float,
    out_path: Path,
) -> int:
    """Draw bounding boxes on an image and save it.

    Args:
        image_path:  Source image.
        boxes:       List of ``[x1, y1, x2, y2]`` boxes in pixel coords.
        scores:      Corresponding confidence scores.
        labels:      Corresponding integer class IDs.
        class_names: Ordered list of class name strings.
        threshold:   Only draw boxes with score ≥ threshold.
        out_path:    Where to save the annotated image.

    Returns:
        Number of detections drawn.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")

    drawn = 0
    for box, score, label_id in zip(boxes, scores, labels):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box
        colour = _BOX_COLOURS[int(label_id) % len(_BOX_COLOURS)]
        name   = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)

        rect = mpatches.FancyBboxPatch(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=colour,
            facecolor="none",
            boxstyle="square,pad=0",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 4, 0),
            f"{name} {score:.2f}",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=colour, alpha=0.7, pad=1, edgecolor="none"),
        )
        drawn += 1

    ax.set_title(f"{image_path.name}  ({drawn} detections above {threshold:.0%})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return drawn


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(
    model,
    postprocess,
    image_paths: list[Path],
    class_names: list[str],
    threshold: float,
    out_dir: Path,
    device: str,
) -> None:
    """Run inference on a list of images and save visualisations.

    Args:
        model:        Loaded D-FINE model.
        postprocess:  D-FINE post-processor.
        image_paths:  List of image paths to process.
        class_names:  Ordered class name strings.
        threshold:    Confidence score threshold.
        out_dir:      Directory to save annotated images.
        device:       Torch device string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        tensor, orig_size = preprocess_image(img_path, device)
        orig_size_tensor  = torch.tensor([orig_size], device=device)

        with torch.no_grad():
            outputs = model(tensor)

        results = postprocess(outputs, orig_size_tensor)[0]

        boxes  = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results["labels"].cpu().tolist()

        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        n = draw_detections(
            img_path, boxes, scores, labels, class_names, threshold, out_path
        )
        print(f"  {img_path.name}  →  {n} detections  →  {out_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    root = dfine_root()

    p = argparse.ArgumentParser(
        description="Run D-FINE inference and save annotated images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pth checkpoint.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=root / "custom" / "configs" / "dfine_m_traffic.yml",
        help="YAML config (must match the checkpoint's architecture).",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a single image file or a directory of images.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "custom" / "predictions",
        help="Directory to save annotated output images.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence score threshold for displaying detections.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda / cpu).",
    )
    p.add_argument(
        "--yolo-dataset",
        type=Path,
        default=root.parent / "yolo_dataset",
        help="YOLO dataset root — used to read class names from dataset.yaml.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (useful for quick tests).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = dfine_root()

    assert_dfine_cloned(root)

    # ── Collect images ────────────────────────────────────────────────────────
    input_path = args.input.resolve()
    if input_path.is_dir():
        image_paths = sorted(
            list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        )
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not image_paths:
        raise ValueError(f"No .jpg/.png images found in {input_path}")

    if args.max_images:
        image_paths = image_paths[: args.max_images]

    # ── Class names ───────────────────────────────────────────────────────────
    yaml_data   = load_dataset_yaml(args.yolo_dataset.resolve() / "dataset.yaml")
    class_names = get_class_names(yaml_data)

    print(f"Device      : {args.device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Classes     : {class_names}")
    print(f"Threshold   : {args.threshold}")
    print(f"Images      : {len(image_paths)}")
    print(f"Output dir  : {args.out_dir}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, postprocess = load_model(
        config_path     = args.config.resolve(),
        checkpoint_path = args.checkpoint.resolve(),
        device          = args.device,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    run_inference(
        model        = model,
        postprocess  = postprocess,
        image_paths  = image_paths,
        class_names  = class_names,
        threshold    = args.threshold,
        out_dir      = args.out_dir.resolve(),
        device       = args.device,
    )

    print(f"\nDone. Annotated images saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
