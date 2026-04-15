"""
Download official pre-trained D-FINE weights from the Peterande/storage releases.

Model weights are saved to ``custom/weights/`` inside the D-FINE folder.

Available model variants and their COCO pre-trained checkpoints:

    N  →  dfine_n_coco.pth    (~16 MB)
    S  →  dfine_s_coco.pth    (~40 MB)
    M  →  dfine_m_coco.pth    (~76 MB)   ← default
    L  →  dfine_l_coco.pth    (~124 MB)
    X  →  dfine_x_coco.pth    (~248 MB)

Objects365 + COCO pretrained (best generalisation for fine-tuning):

    S  →  dfine_s_obj2coco.pth
    M  →  dfine_m_obj2coco.pth           ← recommended for custom datasets
    L  →  dfine_l_obj2coco_e25.pth
    X  →  dfine_x_obj2coco.pth

Usage (from D-FINE/ folder):
    python download_dfine_weights.py               # download D-FINE-M COCO
    python download_dfine_weights.py --model m --pretrain obj2coco
    python download_dfine_weights.py --model l --pretrain coco
    python download_dfine_weights.py --out-dir /scratch/my_weights
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from utils import dfine_root

# ── Release metadata ──────────────────────────────────────────────────────────

BASE_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

# Mapping: (model_size, pretrain_set) → filename on the releases page
_WEIGHT_FILES: dict[tuple[str, str], str] = {
    ("n", "coco"):     "dfine_n_coco.pth",
    ("s", "coco"):     "dfine_s_coco.pth",
    ("m", "coco"):     "dfine_m_coco.pth",
    ("l", "coco"):     "dfine_l_coco.pth",
    ("x", "coco"):     "dfine_x_coco.pth",
    ("s", "obj2coco"): "dfine_s_obj2coco.pth",
    ("m", "obj2coco"): "dfine_m_obj2coco.pth",
    ("l", "obj2coco"): "dfine_l_obj2coco_e25.pth",
    ("x", "obj2coco"): "dfine_x_obj2coco.pth",
    ("s", "obj365"):   "dfine_s_obj365.pth",
    ("m", "obj365"):   "dfine_m_obj365.pth",
    ("l", "obj365"):   "dfine_l_obj365_e25.pth",
    ("x", "obj365"):   "dfine_x_obj365.pth",
}


# ── Download helper ───────────────────────────────────────────────────────────

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple progress printer for :func:`urllib.request.urlretrieve`."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb  = downloaded / 1_048_576
        tot = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {tot:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def download_weights(
    model: str,
    pretrain: str,
    out_dir: Path,
    force: bool = False,
) -> Path:
    """Download the requested D-FINE checkpoint.

    Args:
        model:    Model size letter – one of ``n s m l x``.
        pretrain: Pre-training dataset – ``"coco"``, ``"obj2coco"``, or ``"obj365"``.
        out_dir:  Directory where the ``.pth`` file will be saved.
        force:    Re-download even if the file already exists.

    Returns:
        Local path to the downloaded checkpoint.

    Raises:
        ValueError: If the requested (model, pretrain) combo is not in the table.
        RuntimeError: If the download fails.
    """
    key = (model.lower(), pretrain.lower())
    if key not in _WEIGHT_FILES:
        available = "\n  ".join(f"{k}" for k in sorted(_WEIGHT_FILES))
        raise ValueError(
            f"No weight file registered for model='{model}', pretrain='{pretrain}'.\n"
            f"Available combinations:\n  {available}"
        )

    filename = _WEIGHT_FILES[key]
    url      = f"{BASE_URL}/{filename}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"Checkpoint already exists: {dest}  ({size_mb:.1f} MB) — skipping download.")
        print("Pass --force to re-download.")
        return dest

    print(f"Downloading {filename}")
    print(f"  URL  : {url}")
    print(f"  Dest : {dest}")

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    except Exception as exc:
        # Remove partial download
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc

    print(f"\nSaved to {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download official D-FINE pre-trained weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["n", "s", "m", "l", "x"],
        default="m",
        help="D-FINE model size.",
    )
    p.add_argument(
        "--pretrain",
        choices=["coco", "obj2coco", "obj365"],
        default="obj2coco",
        help=(
            "Pre-training dataset for the checkpoint. "
            "'obj2coco' gives the best fine-tuning starting point for custom datasets."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=dfine_root() / "custom" / "weights",
        help="Directory to save the downloaded checkpoint.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the checkpoint already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = download_weights(
        model    = args.model,
        pretrain = args.pretrain,
        out_dir  = args.out_dir.resolve(),
        force    = args.force,
    )
    print(f"\nCheckpoint ready at:\n  {dest}")
    print("\nTo use for fine-tuning pass this path with the -t flag:")
    print(f"  python train.py -c <config.yml> -t {dest}")


if __name__ == "__main__":
    main()
