"""
Evaluate a fine-tuned (or pre-trained) D-FINE model on val or test split.

Calls the official D-FINE ``train.py --test-only`` which runs the COCO
evaluator and prints AP / AP50 / AP75 and per-category results.

Usage (from D-FINE/ folder):
    # Evaluate the best checkpoint on the val split (default)
    python eval_dfine.py --checkpoint output/dfine_m_traffic/best.pth

    # Evaluate on the test split (prints AP but no ground-truth IoU match
    # is available unless you have test annotations)
    python eval_dfine.py --checkpoint output/dfine_m_traffic/best.pth --split test

    # Use a specific config (e.g. if you changed dataset paths)
    python eval_dfine.py --checkpoint best.pth --config custom/configs/dfine_m_traffic.yml

    # Dry-run: just print the command
    python eval_dfine.py --checkpoint best.pth --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from utils import assert_dfine_cloned, dfine_root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_test_config(base_config: Path, dataset_dir: Path, split: str) -> Path:
    """Write a temporary config that overrides val_dataloader to point at *split*.

    When ``split == "val"`` the base config is returned unchanged.
    For ``split == "test"`` we need to redirect val_dataloader to the test set.

    The override is written to a temporary YAML file that ``__include__``s the
    base config and only changes the relevant paths.

    Args:
        base_config:  Path to the main training config.
        dataset_dir:  Root of the COCO dataset (contains annotations/ and images/).
        split:        ``"val"`` or ``"test"``.

    Returns:
        Path to the config to pass to train.py (the base config or a temp file).
    """
    if split == "val":
        return base_config

    ann_file  = dataset_dir / "annotations" / f"instances_{split}.json"
    img_dir   = dataset_dir / "images" / split

    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found for split '{split}': {ann_file}\n"
            "Run prepare_dfine_dataset.py first."
        )

    # Write a minimal override config
    override_yaml = (
        f"__include__: ['{base_config}']\n\n"
        f"val_dataloader:\n"
        f"  dataset:\n"
        f"    img_folder: {img_dir}\n"
        f"    ann_file: {ann_file}\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="dfine_eval_"
    )
    tmp.write(override_yaml)
    tmp.flush()
    return Path(tmp.name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    root = dfine_root()

    p = argparse.ArgumentParser(
        description="Evaluate a D-FINE checkpoint with the COCO evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pth checkpoint to evaluate.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=root / "custom" / "configs" / "dfine_m_traffic.yml",
        help="YAML config used during training.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "custom" / "dataset",
        help="Root of the prepared COCO dataset.",
    )
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate on.",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of GPUs.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=7778,
        help="Master port (avoid clashing with a running training job).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = dfine_root()

    assert_dfine_cloned(root)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Possibly redirect val_dataloader for test-split evaluation
    config = _make_test_config(
        base_config = args.config.resolve(),
        dataset_dir = args.dataset_dir.resolve(),
        split       = args.split,
    )

    cmd = [
        "uv", "run", "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        f"--master_port={args.port}",
        str(root / "train.py"),
        "-c", str(config),
        "--test-only",
        "-r", str(checkpoint),
    ]

    print(f"Evaluating on split : {args.split}")
    print(f"Checkpoint          : {checkpoint}")
    print(f"Config              : {config}")
    print()
    print("Command:")
    print("  " + " ".join(str(t) for t in cmd))
    print()

    if args.dry_run:
        print("Dry run — exiting.")
        return

    result = subprocess.run(cmd, cwd=str(root), env=os.environ.copy())

    # Clean up temp config if we created one
    if args.split != "val" and config != args.config.resolve():
        Path(config).unlink(missing_ok=True)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
