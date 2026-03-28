"""
Fine-tune D-FINE on a custom dataset.

This is a thin wrapper around the official D-FINE ``train.py``.  It:
1. Validates that the D-FINE repo has been cloned.
2. Validates that the dataset has been prepared (COCO JSON files exist).
3. Validates that the pre-trained weights exist (or offers to download them).
4. Builds and executes the ``torchrun`` training command.

All training hyper-parameters live in the YAML config; this script only handles
the command-line plumbing.

Usage (from D-FINE/ folder):
    # Single GPU (most common for fine-tuning)
    python train_dfine.py

    # Multi-GPU on one node
    python train_dfine.py --nproc 4

    # Resume an interrupted run
    python train_dfine.py --resume output/dfine_m_traffic/checkpoint0050.pth

    # Override config values on the fly
    python train_dfine.py --override epoches=36 optimizer.lr=0.0001
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils import assert_dfine_cloned, dfine_root


# ── Validation helpers ────────────────────────────────────────────────────────

def _check_dataset(dataset_dir: Path) -> None:
    """Fail early if the COCO dataset layout is incomplete."""
    required = [
        dataset_dir / "annotations" / "instances_train.json",
        dataset_dir / "annotations" / "instances_val.json",
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is not fully prepared. Missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nRun first:\n  python prepare_dfine_dataset.py"
        )


def _check_or_download_weights(weights_path: Path, model: str, pretrain: str) -> None:
    """Offer to download weights if the file is absent."""
    if weights_path.exists():
        return

    print(f"Pre-trained weights not found: {weights_path}")
    answer = input("Download now? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        from download_dfine_weights import download_weights
        download_weights(
            model    = model,
            pretrain = pretrain,
            out_dir  = weights_path.parent,
        )
    else:
        print(
            "Aborted. Download manually with:\n"
            f"  python download_dfine_weights.py --model {model} --pretrain {pretrain}"
        )
        sys.exit(1)


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(
    config:    Path,
    weights:   Path | None,
    resume:    Path | None,
    nproc:     int,
    port:      int,
    use_amp:   bool,
    seed:      int,
    overrides: list[str],
    root:      Path,
) -> list[str]:
    """Assemble the ``torchrun`` training command.

    Args:
        config:    Path to the YAML config file.
        weights:   Pre-trained checkpoint for ``-t`` (tune) flag.
        resume:    Checkpoint to resume from (``-r`` flag).
        nproc:     Number of processes / GPUs per node.
        port:      Master port for distributed training.
        use_amp:   Enable automatic mixed precision.
        seed:      Random seed.
        overrides: Extra ``key=value`` strings forwarded to the config.
        root:      D-FINE root directory (where ``train.py`` lives).

    Returns:
        List of command tokens ready for :func:`subprocess.run`.
    """
    cmd = [
        "uv", "run", "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        str(root / "train.py"),
        "-c", str(config),
    ]

    if weights is not None:
        cmd += ["-t", str(weights)]

    if resume is not None:
        cmd += ["-r", str(resume)]

    if use_amp:
        cmd.append("--use-amp")

    cmd += [f"--seed={seed}"]

    # Extra config overrides (passed through after --)
    if overrides:
        cmd += overrides

    return cmd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    root = dfine_root()

    p = argparse.ArgumentParser(
        description="Fine-tune D-FINE-M on the traffic detection dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=root / "custom" / "configs" / "dfine_m_traffic.yml",
        help="Path to the D-FINE YAML configuration file.",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=root / "custom" / "weights" / "dfine_m_obj2coco.pth",
        help="Pre-trained checkpoint to fine-tune from (passed as -t to train.py).",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume an interrupted training run (-r flag).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "custom" / "dataset",
        help="Path to the prepared COCO-format dataset.",
    )
    p.add_argument(
        "--model",
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="D-FINE model size (only used if auto-downloading weights).",
    )
    p.add_argument(
        "--pretrain",
        default="obj2coco",
        choices=["coco", "obj2coco", "obj365"],
        help="Pre-training set for the checkpoint (only used if auto-downloading).",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of GPUs / processes per node.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=7777,
        help="Master port for distributed training.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed-precision training.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run without executing it.",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra config overrides forwarded to train.py, e.g. "
            "--override epoches=72 optimizer.lr=0.00025"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = dfine_root()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    assert_dfine_cloned(root)
    _check_dataset(args.dataset_dir.resolve())

    weights: Path | None = None
    if args.resume is None:
        _check_or_download_weights(
            weights_path = args.weights.resolve(),
            model        = args.model,
            pretrain     = args.pretrain,
        )
        weights = args.weights.resolve()

    config = args.config.resolve()
    if not config.exists():
        raise FileNotFoundError(
            f"Config file not found: {config}\n"
            "Run the setup or check the path."
        )

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = build_command(
        config    = config,
        weights   = weights,
        resume    = args.resume.resolve() if args.resume else None,
        nproc     = args.nproc,
        port      = args.port,
        use_amp   = not args.no_amp,
        seed      = args.seed,
        overrides = args.override or [],
        root      = root,
    )

    print("Training command:")
    print("  " + " ".join(str(t) for t in cmd))
    print()

    if args.dry_run:
        print("Dry run — exiting without launching training.")
        return

    # Run from the D-FINE root so relative paths in configs resolve correctly
    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=str(root), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
