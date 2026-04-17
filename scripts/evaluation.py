#!/usr/bin/env python3
"""Unified evaluation script for YOLO, Swin-Transformer, RCNN, and D-FINE.

The script:
1) Loads all four model checkpoints from project-relative paths.
2) Runs inference on ``data_final/images/test``.
3) Computes COCO-style detection metrics from YOLO-format labels:
   - mAP@0.50:0.95
   - mAP@0.50
   - mAP@0.75
   - precision / recall / F1 at IoU 0.50
4) Measures inference speed (images/sec and ms/image).
5) Saves plots and a JSON summary under ``results/``.

Notes:
- Paths are resolved relative to repository root to stay portable across machines.
- D-FINE is evaluated via its official ``train.py --test-only`` CLI and parsed from
  stdout. If that environment is unavailable, the script reports the failure while
  still evaluating the remaining models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF


@dataclass
class DetectionRecord:
    boxes: torch.Tensor  # [N, 4], xyxy absolute
    scores: torch.Tensor  # [N]
    labels: torch.Tensor  # [N], 1-indexed class labels


@dataclass
class EvalMetrics:
    map_50_95: float
    map_50: float
    map_75: float
    precision_50: float
    recall_50: float
    f1_50: float
    images_per_second: float
    ms_per_image: float
    num_images: int
    error: str | None = None


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_images(images_dir: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files: list[Path] = []
    for ext in exts:
        files.extend(images_dir.glob(f"*{ext}"))
    files = sorted(set(files), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return files


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def filter_readable_images(paths: list[Path]) -> list[Path]:
    valid, bad = [], []
    for p in paths:
        (valid if _is_readable_image(p) else bad).append(p)

    if bad:
        sample = ", ".join(x.name for x in bad[:5])
        more = f" ... (+{len(bad) - 5} more)" if len(bad) > 5 else ""
        print(f"Warning: skipped {len(bad)} unreadable image(s): {sample}{more}")

    if not valid:
        raise RuntimeError("No readable test images found after filtering corrupted files.")
    return valid


def parse_yolo_labels(label_path: Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    boxes, labels = [], []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0]) + 1
                xc, yc, bw, bh = (float(v) for v in parts[1:])
                x1 = max(0.0, (xc - bw / 2.0) * w)
                y1 = max(0.0, (yc - bh / 2.0) * h)
                x2 = min(float(w), (xc + bw / 2.0) * w)
                y2 = min(float(h), (yc + bh / 2.0) * h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def build_ground_truth(test_images: list[Path], labels_dir: Path) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    for img_path in test_images:
        with Image.open(img_path) as img:
            w, h = img.size
        lab_path = labels_dir / f"{img_path.stem}.txt"
        boxes, labels = parse_yolo_labels(lab_path, w, h)
        records.append(
            DetectionRecord(
                boxes=boxes,
                labels=labels,
                scores=torch.ones((labels.numel(),), dtype=torch.float32),
            )
        )
    return records


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _per_iou_map(
    preds: list[DetectionRecord],
    gts: list[DetectionRecord],
    iou_threshold: float,
) -> tuple[float, float, float]:
    class_ids = sorted({int(lbl.item()) for g in gts for lbl in g.labels})
    if not class_ids:
        return 0.0, 0.0, 0.0

    ap_list, p_list, r_list = [], [], []
    for cls in class_ids:
        n_gt = 0
        pred_entries: list[tuple[float, int, torch.Tensor]] = []
        gt_per_img: dict[int, torch.Tensor] = {}
        matched: dict[int, torch.Tensor] = {}

        for i, (pred_i, gt_i) in enumerate(zip(preds, gts)):
            gt_mask = gt_i.labels == cls
            gt_boxes = gt_i.boxes[gt_mask]
            gt_per_img[i] = gt_boxes
            matched[i] = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
            n_gt += gt_boxes.shape[0]

            pd_mask = pred_i.labels == cls
            pd_boxes = pred_i.boxes[pd_mask]
            pd_scores = pred_i.scores[pd_mask]
            for b, s in zip(pd_boxes, pd_scores):
                pred_entries.append((float(s.item()), i, b))

        if n_gt == 0:
            continue

        pred_entries.sort(key=lambda x: x[0], reverse=True)
        if not pred_entries:
            ap_list.append(0.0)
            p_list.append(0.0)
            r_list.append(0.0)
            continue

        tp = np.zeros((len(pred_entries),), dtype=np.float32)
        fp = np.zeros((len(pred_entries),), dtype=np.float32)
        for k, (_, img_id, box) in enumerate(pred_entries):
            gt_boxes = gt_per_img[img_id]
            if gt_boxes.shape[0] == 0:
                fp[k] = 1.0
                continue
            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            if best_iou.item() >= iou_threshold and not matched[img_id][best_idx]:
                tp[k] = 1.0
                matched[img_id][best_idx] = True
            else:
                fp[k] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(float(n_gt), 1e-8)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
        ap = _compute_ap(recall, precision) if recall.size else 0.0

        ap_list.append(ap)
        p_list.append(float(precision[-1]) if precision.size else 0.0)
        r_list.append(float(recall[-1]) if recall.size else 0.0)

    if not ap_list:
        return 0.0, 0.0, 0.0
    return float(np.mean(ap_list)), float(np.mean(p_list)), float(np.mean(r_list))


def compute_detection_metrics(preds: list[DetectionRecord], gts: list[DetectionRecord]) -> dict[str, float]:
    ious = np.arange(0.5, 1.0, 0.05)
    ap_by_iou = []
    p50, r50, ap50 = 0.0, 0.0, 0.0
    ap75 = 0.0

    for thr in ious:
        ap, p, r = _per_iou_map(preds, gts, float(thr))
        ap_by_iou.append(ap)
        if math.isclose(thr, 0.5, abs_tol=1e-6):
            ap50, p50, r50 = ap, p, r
        if math.isclose(thr, 0.75, abs_tol=1e-6):
            ap75 = ap

    f1_50 = (2 * p50 * r50 / (p50 + r50)) if (p50 + r50) > 0 else 0.0
    return {
        "mAP_50_95": float(np.mean(ap_by_iou)) if ap_by_iou else 0.0,
        "mAP_50": ap50,
        "mAP_75": ap75,
        "precision_50": p50,
        "recall_50": r50,
        "f1_50": f1_50,
    }


def infer_with_yolo(model_path: Path, test_images: list[Path], device: torch.device) -> tuple[list[DetectionRecord], float]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    preds: list[DetectionRecord] = []

    t0 = time.perf_counter()
    for img_path in test_images:
        # Align YOLO confidence filtering with torchvision models (score threshold = 0.05).
        result = model.predict(
            source=str(img_path),
            verbose=False,
            device=0 if device.type == "cuda" else "cpu",
            conf=0.05,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu() if result.boxes is not None else torch.zeros((0, 4))
        scores = result.boxes.conf.detach().cpu() if result.boxes is not None else torch.zeros((0,))
        labels = result.boxes.cls.detach().cpu().to(torch.int64) + 1 if result.boxes is not None else torch.zeros((0,), dtype=torch.int64)
        preds.append(DetectionRecord(boxes=boxes, scores=scores, labels=labels))
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def infer_with_torchvision_model(
    model: torch.nn.Module,
    test_images: list[Path],
    device: torch.device,
) -> tuple[list[DetectionRecord], float]:
    model.eval()
    preds: list[DetectionRecord] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for img_path in test_images:
            with Image.open(img_path) as img:
                tensor = TF.to_tensor(img.convert("RGB")).to(device)
            out = model([tensor])[0]
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu().to(torch.int64)
            preds.append(DetectionRecord(boxes=boxes, scores=scores, labels=labels))
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def evaluate_yolo(
    model_path: Path,
    test_images: list[Path],
    gts: list[DetectionRecord],
    device: torch.device,
) -> EvalMetrics:
    preds, elapsed = infer_with_yolo(model_path, test_images, device)
    m = compute_detection_metrics(preds, gts)
    ips = len(test_images) / max(elapsed, 1e-8)
    return EvalMetrics(
        map_50_95=m["mAP_50_95"],
        map_50=m["mAP_50"],
        map_75=m["mAP_75"],
        precision_50=m["precision_50"],
        recall_50=m["recall_50"],
        f1_50=m["f1_50"],
        images_per_second=ips,
        ms_per_image=1000.0 / ips,
        num_images=len(test_images),
    )


def evaluate_rcnn(
    model_path: Path,
    test_images: list[Path],
    gts: list[DetectionRecord],
    device: torch.device,
    project_root: Path,
) -> EvalMetrics:
    rcnn_dir = project_root / "models" / "rcnn"
    if str(rcnn_dir) not in sys.path:
        sys.path.insert(0, str(rcnn_dir))

    from rcnn_train import get_model  # type: ignore

    model = get_model(num_classes=3)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    preds, elapsed = infer_with_torchvision_model(model, test_images, device)
    m = compute_detection_metrics(preds, gts)
    ips = len(test_images) / max(elapsed, 1e-8)
    return EvalMetrics(
        map_50_95=m["mAP_50_95"],
        map_50=m["mAP_50"],
        map_75=m["mAP_75"],
        precision_50=m["precision_50"],
        recall_50=m["recall_50"],
        f1_50=m["f1_50"],
        images_per_second=ips,
        ms_per_image=1000.0 / ips,
        num_images=len(test_images),
    )


def evaluate_swin(
    model_path: Path,
    test_images: list[Path],
    gts: list[DetectionRecord],
    device: torch.device,
    project_root: Path,
) -> EvalMetrics:
    swin_dir = project_root / "models" / "swin_transformer"
    if str(swin_dir) not in sys.path:
        sys.path.insert(0, str(swin_dir))

    from swinTransformer import TrainConfig, build_model  # type: ignore

    model = build_model(TrainConfig())
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    preds, elapsed = infer_with_torchvision_model(model, test_images, device)
    m = compute_detection_metrics(preds, gts)
    ips = len(test_images) / max(elapsed, 1e-8)
    return EvalMetrics(
        map_50_95=m["mAP_50_95"],
        map_50=m["mAP_50"],
        map_75=m["mAP_75"],
        precision_50=m["precision_50"],
        recall_50=m["recall_50"],
        f1_50=m["f1_50"],
        images_per_second=ips,
        ms_per_image=1000.0 / ips,
        num_images=len(test_images),
    )


def _parse_dfine_stdout(text: str) -> tuple[float, float, float]:
    """Parse COCO AP lines; value is after ``] =``, not ``IoU=0.50``."""
    map_50_95: float | None = None
    map_50: float | None = None
    map_75: float | None = None

    for line in text.splitlines():
        if "Average Precision" not in line or "(AP)" not in line:
            continue
        if "area=" not in line or "all" not in line:
            continue
        tail = re.search(r"\]\s*=\s*([0-9.]+)\s*$", line.strip())
        if not tail:
            continue
        val = float(tail.group(1))
        if "IoU=0.50:0.95" in line:
            map_50_95 = val
        elif "IoU=0.50" in line and "0.50:0.95" not in line:
            map_50 = val
        elif "IoU=0.75" in line:
            map_75 = val

    if map_50_95 is not None and map_50 is not None and map_75 is not None:
        return map_50_95, map_50, map_75
    raise RuntimeError(
        "Could not parse D-FINE AP/AP50/AP75 from evaluator output "
        f"(got mAP5095={map_50_95}, mAP50={map_50}, mAP75={map_75})."
    )


def evaluate_dfine(
    checkpoint_path: Path,
    project_root: Path,
    num_images: int,
    dfine_config: str | None,
) -> EvalMetrics:
    dfine_root = project_root / "models" / "D-FINE"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=7788",
        "train.py",
    ]

    if dfine_config:
        command.extend(["-c", dfine_config])
    else:
        command.extend(["-c", "custom/configs/dfine_m_traffic.yml"])

    command.extend(["--test-only", "-r", str(checkpoint_path)])
    t0 = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(dfine_root),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "D-FINE evaluation failed").strip()
        raise RuntimeError(err)

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    map_50_95, map_50, map_75 = _parse_dfine_stdout(combined)
    ips = num_images / max(elapsed, 1e-8)
    return EvalMetrics(
        map_50_95=map_50_95,
        map_50=map_50,
        map_75=map_75,
        precision_50=float("nan"),
        recall_50=float("nan"),
        f1_50=float("nan"),
        images_per_second=ips,
        ms_per_image=1000.0 / ips,
        num_images=num_images,
    )


def make_bar_plot(
    names: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    invert: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    bars = ax.bar(names, values, color=colors[: len(names)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    finite_values = [v for v in values if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if finite_values:
        ymax = max(finite_values) * (1.2 if not invert else 1.1)
        ymin = 0.0 if not invert else max(0.0, min(finite_values) * 0.9)
        ax.set_ylim(ymin, max(ymax, ymin + 1e-6))

    for b, v in zip(bars, values):
        label = "N/A" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + (0.01 * (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1)),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_overall_plot(results: dict[str, EvalMetrics], output_path: Path) -> None:
    # Overall score balances accuracy and speed (speed normalized by best model).
    names = list(results.keys())
    speed = np.array([results[n].images_per_second for n in names], dtype=float)
    map5095 = np.array([results[n].map_50_95 for n in names], dtype=float)
    map50 = np.array([results[n].map_50 for n in names], dtype=float)
    map75 = np.array([results[n].map_75 for n in names], dtype=float)

    speed_norm = speed / max(np.max(speed), 1e-8)
    overall = 0.4 * map5095 + 0.25 * map50 + 0.25 * map75 + 0.10 * speed_norm

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, map5095, marker="o", linewidth=2, label="mAP@0.50:0.95")
    ax.plot(x, map50, marker="o", linewidth=2, label="mAP@0.50")
    ax.plot(x, map75, marker="o", linewidth=2, label="mAP@0.75")
    ax.plot(x, overall, marker="D", linestyle="--", linewidth=2, label="Overall score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Overall model comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def safe_eval(
    fn: Callable[[], EvalMetrics],
    model_name: str,
) -> EvalMetrics:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return EvalMetrics(
            map_50_95=float("nan"),
            map_50=float("nan"),
            map_75=float("nan"),
            precision_50=float("nan"),
            recall_50=float("nan"),
            f1_50=float("nan"),
            images_per_second=float("nan"),
            ms_per_image=float("nan"),
            num_images=0,
            error=f"{model_name}: {exc}",
        )


def to_serializable(metrics: EvalMetrics) -> dict:
    return {
        "mAP_50_95": metrics.map_50_95,
        "mAP_50": metrics.map_50,
        "mAP_75": metrics.map_75,
        "precision_50": metrics.precision_50,
        "recall_50": metrics.recall_50,
        "f1_50": metrics.f1_50,
        "images_per_second": metrics.images_per_second,
        "ms_per_image": metrics.ms_per_image,
        "num_images": metrics.num_images,
        "error": metrics.error,
    }


def parse_args() -> argparse.Namespace:
    root = resolve_project_root()
    p = argparse.ArgumentParser(description="Evaluate all traffic detection models on test split.")
    p.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    p.add_argument("--test-images", type=Path, default=root / "data_final" / "images" / "test")
    p.add_argument("--test-labels", type=Path, default=root / "data_final" / "labels" / "test")
    p.add_argument("--plots-dir", type=Path, default=root / "results")
    p.add_argument("--dfine-config", default=None, help="Optional D-FINE config path relative to D-FINE root.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_paths = {
        "YOLO (Baseline)": root / "models" / "yolo" / "yolo_untuned.pt",
        "Swin Transformer": root / "models" / "swin_transformer" / "best_model.pt",
        "RCNN": root / "models" / "rcnn" / "rcnn_untuned.pt",
        "D-FINE": root / "models" / "D-FINE" / "output" / "dfine_m_traffic" / "best_stg1.pth",
    }

    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {model_path}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    test_images = filter_readable_images(list_images(args.test_images))
    gts = build_ground_truth(test_images, args.test_labels)

    print(f"Device: {device}")
    print(f"Test images: {len(test_images)} ({args.test_images})")
    print(f"Saving plots/results to: {args.plots_dir}")

    results: dict[str, EvalMetrics] = {}
    results["YOLO (Baseline)"] = safe_eval(
        lambda: evaluate_yolo(model_paths["YOLO (Baseline)"], test_images, gts, device),
        "YOLO (Baseline)",
    )
    results["Swin Transformer"] = safe_eval(
        lambda: evaluate_swin(model_paths["Swin Transformer"], test_images, gts, device, root),
        "Swin Transformer",
    )
    results["RCNN"] = safe_eval(
        lambda: evaluate_rcnn(model_paths["RCNN"], test_images, gts, device, root),
        "RCNN",
    )
    results["D-FINE"] = safe_eval(
        lambda: evaluate_dfine(
            model_paths["D-FINE"],
            root,
            num_images=len(test_images),
            dfine_config=args.dfine_config,
        ),
        "D-FINE",
    )

    names = list(results.keys())
    map_50_95_values = [results[n].map_50_95 for n in names]
    map_50_values = [results[n].map_50 for n in names]
    map_75_values = [results[n].map_75 for n in names]
    speed_values = [results[n].images_per_second for n in names]

    make_bar_plot(
        names,
        map_50_95_values,
        "mAP@0.50:0.95 Comparison",
        "mAP@0.50:0.95",
        args.plots_dir / "map_50_95_comparison.png",
    )
    make_bar_plot(
        names,
        map_50_values,
        "mAP@0.50 Comparison",
        "mAP@0.50",
        args.plots_dir / "map_50_comparison.png",
    )
    make_bar_plot(
        names,
        map_75_values,
        "mAP@0.75 Comparison",
        "mAP@0.75",
        args.plots_dir / "map_75_comparison.png",
    )
    make_bar_plot(
        names,
        speed_values,
        "Inference Speed Comparison",
        "Images / second",
        args.plots_dir / "inference_speed_comparison.png",
    )
    make_overall_plot(results, args.plots_dir / "overall_performance.png")

    report = {name: to_serializable(metrics) for name, metrics in results.items()}
    with open(args.plots_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for name, m in results.items():
        print(f"\n{name}")
        if m.error:
            print(f"  ERROR: {m.error}")
            continue
        print(f"  mAP@0.50:0.95 : {m.map_50_95:.4f}")
        print(f"  mAP@0.50      : {m.map_50:.4f}")
        print(f"  mAP@0.75      : {m.map_75:.4f}")
        print(f"  Precision@0.5 : {m.precision_50:.4f}")
        print(f"  Recall@0.5    : {m.recall_50:.4f}")
        print(f"  F1@0.5        : {m.f1_50:.4f}")
        print(f"  Speed         : {m.images_per_second:.2f} img/s ({m.ms_per_image:.2f} ms/img)")

    print(f"\nSaved JSON summary: {args.plots_dir / 'evaluation_summary.json'}")
    print("Saved plots:")
    print(f"- {args.plots_dir / 'map_50_95_comparison.png'}")
    print(f"- {args.plots_dir / 'map_50_comparison.png'}")
    print(f"- {args.plots_dir / 'map_75_comparison.png'}")
    print(f"- {args.plots_dir / 'inference_speed_comparison.png'}")
    print(f"- {args.plots_dir / 'overall_performance.png'}")


if __name__ == "__main__":
    main()
