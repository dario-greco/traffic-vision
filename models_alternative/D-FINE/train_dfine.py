"""
Fine-tune D-FINE on a custom dataset.

This is a thin wrapper around the official D-FINE ``train.py``.  It:
1. Validates that the D-FINE repo has been cloned.
2. Validates that the dataset has been prepared (COCO JSON files exist).
3. Validates that the pre-trained weights exist (or offers to download them).
4. Builds and executes the ``torchrun`` training command.

By default this script also:
- Passes ``epochs=5`` to the trainer (override with ``--override epochs=N``).
- Passes ``checkpoint_freq=6`` (extra checkpoint every 6 epochs).
- Forces ``output_dir`` to ``output/dfine_m_traffic`` (relative to the D-FINE root)
  unless you override it.
- If ``output_dir/best_stg2.pth`` or ``output_dir/best_stg1.pth`` exists and you
  did not pass ``--resume``, it resumes from that checkpoint (full solver state).
  Use ``--from-scratch`` to always start from ``--weights`` (-t) instead.

All training hyper-parameters live in the YAML config; this script only handles
the command-line plumbing.

Usage (from D-FINE/ folder):
    # Single GPU (most common for fine-tuning)
    python train_dfine.py

    # Multi-GPU on one node
    python train_dfine.py --nproc 4

    # Resume a specific checkpoint (wins over auto best)
    python train_dfine.py --resume output/dfine_m_traffic/last.pth

    # Override config values on the fly
    python train_dfine.py --override epochs=36 optimizer.lr=0.0001

    # Ignore saved best checkpoints; fine-tune from pretrained -t again
    python train_dfine.py --from-scratch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from utils import assert_dfine_cloned, dfine_root


# Default values forwarded to train.py unless explicitly overridden.
DEFAULT_TRAIN_EPOCHS = 5
DEFAULT_CHECKPOINT_FREQ = 6
DEFAULT_OUTPUT_DIR = Path("output/dfine_m_traffic")


def _output_dir_from_config(config_path: Path, root: Path) -> Path:
    """Resolve ``output_dir`` from the top-level YAML (same key as in traffic config)."""
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    rel = data.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _find_best_checkpoint(output_dir: Path) -> Path | None:
    """Return best saved weights if present (stage 2 preferred over stage 1)."""
    for name in ("best_stg2.pth", "best_stg1.pth"):
        ckpt = output_dir / name
        if ckpt.is_file():
            return ckpt
    return None


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

    # train.py expects ``-u key=value ...`` for YAML updates (see yaml_utils.parse_cli).
    if overrides:
        if overrides[0] in ("-u", "--update"):
            cmd += overrides
        else:
            cmd += ["-u", *overrides]

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
        help=(
            "Checkpoint to resume (-r). If omitted, a best_stg2/best_stg1 under "
            "the config's output_dir is used automatically unless --from-scratch."
        ),
    )
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not auto-resume from best_stg*.pth; use --weights (-t) like a new fine-tune.",
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
            "Extra YAML updates for train.py (-u), e.g. "
            "--override epochs=72 optimizer.lr=0.00025"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = dfine_root()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    assert_dfine_cloned(root)
    _check_dataset(args.dataset_dir.resolve())

    config = args.config.resolve()
    if not config.exists():
        raise FileNotFoundError(
            f"Config file not found: {config}\n"
            "Run the setup or check the path."
        )

    resume_path: Path | None = args.resume.resolve() if args.resume else None
    if resume_path is None and not args.from_scratch:
        auto_best = _find_best_checkpoint(_output_dir_from_config(config, root))
        if auto_best is not None:
            resume_path = auto_best
            print(f"Using best checkpoint (auto): {resume_path}\n")

    weights: Path | None = None
    if resume_path is None:
        _check_or_download_weights(
            weights_path = args.weights.resolve(),
            model        = args.model,
            pretrain     = args.pretrain,
        )
        weights = args.weights.resolve()

    overrides = list(args.override or [])

    # Enforce defaults unless explicitly overridden.
    if not any(o.split("=", 1)[0] == "output_dir" for o in overrides):
        overrides.insert(0, f"output_dir={DEFAULT_OUTPUT_DIR.as_posix()}")
    if not any(o.split("=", 1)[0] == "checkpoint_freq" for o in overrides):
        overrides.insert(0, f"checkpoint_freq={DEFAULT_CHECKPOINT_FREQ}")
    if not any(o.split("=", 1)[0] == "epochs" for o in overrides):
        overrides.insert(0, f"epochs={DEFAULT_TRAIN_EPOCHS}")

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = build_command(
        config    = config,
        weights   = weights,
        resume    = resume_path,
        nproc     = args.nproc,
        port      = args.port,
        use_amp   = not args.no_amp,
        seed      = args.seed,
        overrides = overrides,
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
