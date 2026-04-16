#!/usr/bin/env python3
"""Evaluate only the D-FINE checkpoint on the test split.

This is a lightweight pre-check script so you can validate D-FINE first
before running the full `scripts/evaluation.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def count_test_images(test_images_dir: Path) -> int:
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    n = 0
    for ext in exts:
        n += len(list(test_images_dir.glob(f"*{ext}")))
    if n == 0:
        raise FileNotFoundError(f"No test images found in: {test_images_dir}")
    return n


def parse_dfine_stdout(text: str) -> tuple[float, float, float]:
    """Parse COCO bbox AP lines from D-FINE / pycocotools text output.

    IMPORTANT: Do not use the first ``=`` on the line — that is ``IoU=0.50``.
    The metric value is always after the closing ``]``, e.g.
    ``... maxDets=100 ] = 0.307``.
    """
    map_50_95: float | None = None
    map_50: float | None = None
    map_75: float | None = None

    for line in text.splitlines():
        if "Average Precision" not in line or "(AP)" not in line:
            continue
        # Only the overall row (skip area= small | medium | large)
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
        "Could not parse AP/AP50/AP75 from D-FINE output "
        f"(got mAP5095={map_50_95}, mAP50={map_50}, mAP75={map_75})."
    )


def run_dfine_eval(dfine_root: Path, config: str, checkpoint: Path) -> tuple[str, float]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=7788",
        "train.py",
        "-c",
        config,
        "--test-only",
        "-r",
        str(checkpoint),
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(dfine_root),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "D-FINE eval failed").strip()
        raise RuntimeError(err)
    # Some builds log metrics to stderr; merge for parsing.
    return (proc.stdout or "") + "\n" + (proc.stderr or ""), elapsed


def main() -> None:
    root = resolve_project_root()
    parser = argparse.ArgumentParser(description="Evaluate D-FINE only.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "D-FINE" / "output" / "dfine_m_traffic" / "best_stg1.pth",
        help="Path to D-FINE checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        default="custom/configs/dfine_m_traffic.yml",
        help="D-FINE config path relative to models/D-FINE.",
    )
    parser.add_argument(
        "--test-images",
        type=Path,
        default=root / "data_final" / "images" / "test",
        help="Test image folder (used for speed normalization).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "plots" / "dfine_only_summary.json",
        help="Where to save D-FINE-only metrics JSON.",
    )
    args = parser.parse_args()

    dfine_root = root / "models" / "D-FINE"
    if not dfine_root.exists():
        raise FileNotFoundError(f"D-FINE folder not found: {dfine_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    n_images = count_test_images(args.test_images)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"D-FINE root     : {dfine_root}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Config (relpath): {args.config}")
    print(f"Test images     : {n_images}")
    print("Running D-FINE evaluation...")

    try:
        stdout, elapsed = run_dfine_eval(dfine_root, args.config, args.checkpoint)
        map_50_95, map_50, map_75 = parse_dfine_stdout(stdout)
        img_per_sec = n_images / max(elapsed, 1e-8)
        ms_per_img = 1000.0 / img_per_sec

        result = {
            "status": "ok",
            "mAP_50_95": map_50_95,
            "mAP_50": map_50,
            "mAP_75": map_75,
            "images_per_second": img_per_sec,
            "ms_per_image": ms_per_img,
            "num_images": n_images,
            "elapsed_sec": elapsed,
        }

        print("\n=== D-FINE Evaluation Summary ===")
        print(f"mAP@0.50:0.95 : {map_50_95:.4f}")
        print(f"mAP@0.50      : {map_50:.4f}")
        print(f"mAP@0.75      : {map_75:.4f}")
        print(f"Speed         : {img_per_sec:.2f} img/s ({ms_per_img:.2f} ms/img)")

    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "No module named 'tensorboard'" in msg:
            msg = (
                f"{msg}\nHint: install it with `uv add tensorboard`, then rerun this script."
            )
        result = {"status": "error", "error": msg}
        print("\nD-FINE evaluation failed.")
        print(msg)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output_json}")

    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
