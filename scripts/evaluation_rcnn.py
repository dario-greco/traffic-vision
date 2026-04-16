#!/usr/bin/env python3
"""Evaluate only the fine-tuned RCNN checkpoint on the test split.

This script computes the same metrics as ``scripts/evaluation.py``:
- mAP@0.50:0.95
- mAP@0.50
- mAP@0.75
- precision/recall/F1 at IoU 0.50
- inference speed (images/sec, ms/image)

It saves RCNN-only plots and a JSON summary under ``plots/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluation import (  # type: ignore
    EvalMetrics,
    build_ground_truth,
    evaluate_rcnn,
    filter_readable_images,
    list_images,
    resolve_project_root,
    to_serializable,
)


def _single_bar_plot(
    value: float,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#4C78A8",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bar = ax.bar(["Fine-tuned RCNN"], [value], color=color)[0]
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ymax = max(value * 1.2, 1e-6)
    ax.set_ylim(0.0, ymax)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (0.01 * ymax),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _metric_triplet_plot(
    metrics: EvalMetrics,
    output_path: Path,
) -> None:
    names = ["mAP@0.50:0.95", "mAP@0.50", "mAP@0.75"]
    values = [metrics.map_50_95, metrics.map_50, metrics.map_75]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors)
    ax.set_title("Fine-tuned RCNN mAP metrics")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, max(max(values) * 1.2, 1e-6))
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + (0.01 * ax.get_ylim()[1]),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _prf1_plot(metrics: EvalMetrics, output_path: Path) -> None:
    names = ["Precision@0.50", "Recall@0.50", "F1@0.50"]
    values = [metrics.precision_50, metrics.recall_50, metrics.f1_50]
    colors = ["#72B7B2", "#E45756", "#54A24B"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors)
    ax.set_title("Fine-tuned RCNN Precision/Recall/F1")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, max(max(values) * 1.2, 1e-6))
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + (0.01 * ax.get_ylim()[1]),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = resolve_project_root()
    p = argparse.ArgumentParser(description="Evaluate only the fine-tuned RCNN model.")
    p.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "rcnn" / "FINAL_PRODUCTION_MODEL.pt",
        help="Path to fine-tuned RCNN checkpoint.",
    )
    p.add_argument("--test-images", type=Path, default=root / "data_final" / "images" / "test")
    p.add_argument("--test-labels", type=Path, default=root / "data_final" / "labels" / "test")
    p.add_argument("--plots-dir", type=Path, default=root / "plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing RCNN checkpoint: {args.checkpoint}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    test_images = filter_readable_images(list_images(args.test_images))
    gts = build_ground_truth(test_images, args.test_labels)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test images: {len(test_images)} ({args.test_images})")
    print(f"Saving plots/results to: {args.plots_dir}")

    metrics = evaluate_rcnn(args.checkpoint, test_images, gts, device, root)

    _metric_triplet_plot(metrics, args.plots_dir / "rcnn_map_metrics.png")
    _prf1_plot(metrics, args.plots_dir / "rcnn_prf1_metrics.png")
    _single_bar_plot(
        metrics.images_per_second,
        "Fine-tuned RCNN Inference Speed",
        "Images / second",
        args.plots_dir / "rcnn_inference_speed.png",
        color="#B279A2",
    )
    _single_bar_plot(
        metrics.ms_per_image,
        "Fine-tuned RCNN Latency",
        "Milliseconds / image",
        args.plots_dir / "rcnn_latency_ms_per_image.png",
        color="#FF9DA6",
    )

    report = {"Fine-tuned RCNN": to_serializable(metrics)}
    summary_path = args.plots_dir / "rcnn_only_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Fine-tuned RCNN Evaluation Summary ===")
    print(f"mAP@0.50:0.95 : {metrics.map_50_95:.4f}")
    print(f"mAP@0.50      : {metrics.map_50:.4f}")
    print(f"mAP@0.75      : {metrics.map_75:.4f}")
    print(f"Precision@0.5 : {metrics.precision_50:.4f}")
    print(f"Recall@0.5    : {metrics.recall_50:.4f}")
    print(f"F1@0.5        : {metrics.f1_50:.4f}")
    print(f"Speed         : {metrics.images_per_second:.2f} img/s ({metrics.ms_per_image:.2f} ms/img)")
    print(f"\nSaved JSON summary: {summary_path}")
    print("Saved plots:")
    print(f"- {args.plots_dir / 'rcnn_map_metrics.png'}")
    print(f"- {args.plots_dir / 'rcnn_prf1_metrics.png'}")
    print(f"- {args.plots_dir / 'rcnn_inference_speed.png'}")
    print(f"- {args.plots_dir / 'rcnn_latency_ms_per_image.png'}")


if __name__ == "__main__":
    main()
