"""
Shared utilities for the D-FINE traffic-detection pipeline.

All scripts in this folder import from here to avoid duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


# ── Path helpers ──────────────────────────────────────────────────────────────

def dfine_root() -> Path:
    """Absolute path to the D-FINE/ folder (parent of this file)."""
    return Path(__file__).parent.resolve()


def repo_root() -> Path:
    """Absolute path to the traffic-vision repo root (parent of D-FINE/)."""
    return dfine_root().parent


# ── dataset.yaml helpers ──────────────────────────────────────────────────────

def load_dataset_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load and minimally validate a YOLO dataset.yaml file.

    Args:
        yaml_path: Path to dataset.yaml.

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Required ``names`` key is absent.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at: {yaml_path}\n"
            "Make sure the YOLO dataset has been prepared (run scripts/0_prepare_data.py)."
        )

    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    if "names" not in data:
        raise ValueError(
            "dataset.yaml is missing the required 'names' field.\n"
            "Expected format:\n"
            "  names:\n"
            "    0: stop sign\n"
            "    1: traffic light"
        )

    return data


def get_class_names(dataset_yaml: dict[str, Any]) -> list[str]:
    """Return an ordered list of class names from a parsed dataset.yaml.

    Handles both dict (``{0: 'cls', 1: 'cls'}``) and list formats.

    Args:
        dataset_yaml: Output of :func:`load_dataset_yaml`.

    Returns:
        Class names sorted by class ID.
    """
    names = dataset_yaml["names"]

    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]

    if isinstance(names, list):
        return list(names)

    raise ValueError(
        f"Unexpected type for 'names' in dataset.yaml: {type(names)}. "
        "Expected dict or list."
    )


# ── Image helpers ─────────────────────────────────────────────────────────────

def image_size(image_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of an image without decoding pixel data.

    Uses :mod:`PIL` for format-agnostic reading.

    Args:
        image_path: Path to a JPEG or PNG image.

    Returns:
        ``(width, height)`` in pixels.
    """
    from PIL import Image  # deferred import keeps startup fast

    with Image.open(image_path) as img:
        return img.size  # PIL returns (width, height)


# ── D-FINE repo validation ────────────────────────────────────────────────────

def assert_dfine_cloned(root: Path | None = None) -> None:
    """Raise if this training tree is incomplete (core files missing).

    Args:
        root: D-FINE root directory. Defaults to :func:`dfine_root`.

    Raises:
        RuntimeError: If ``train.py``, ``src/``, or ``configs/`` are absent.
    """
    root = root or dfine_root()
    sentinel_paths = [root / "train.py", root / "src", root / "configs"]
    missing = [p for p in sentinel_paths if not p.exists()]

    if missing:
        raise RuntimeError(
            "D-FINE training files are incomplete or this is not the D-FINE root.\n"
            f"Missing: {[str(p.relative_to(root)) for p in missing]}\n\n"
            "Expected ``train.py``, ``src/``, and ``configs/`` next to these scripts.\n"
            "Install deps with: pip install -r requirements.txt\n"
        )


def add_dfine_src_to_path(root: Path | None = None) -> None:
    """Prepend the D-FINE ``src/`` directory to :data:`sys.path`.

    Necessary for importing D-FINE internals (e.g. for inference).

    Args:
        root: D-FINE root directory. Defaults to :func:`dfine_root`.
    """
    root = root or dfine_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    # Also add the D-FINE root itself so ``import dfine`` style imports work
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
