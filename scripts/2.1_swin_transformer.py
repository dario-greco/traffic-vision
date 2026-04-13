#!/usr/bin/env python3
"""Train Swin-T + FPN + RetinaNet on a YOLO-format traffic dataset.

Expects ``<data_dir>/images/{train,val,test}`` and matching ``labels/`` with
``class xc yc w h`` lines (normalised). Classes 0–1 map to RetinaNet labels 1–2
(background is 0).

Training stack: mixed precision (CUDA), gradient accumulation, cosine LR with
warmup, optional periodic full train-set mAP (costly), and separate inference
batch size. Checkpointing uses best validation mAP@0.5.

CLI examples::

    python scripts/2.1_swin_transformer.py
    python scripts/2.1_swin_transformer.py --sweep --epochs 25
    python scripts/2.1_swin_transformer.py --preset freeze_less
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "swin_runs"

CLASS_NAMES = {0: "stop sign", 1: "traffic light"}
NUM_CLASSES = len(CLASS_NAMES) + 1

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class TrainConfig:
    """One training run. ``freeze_stages`` freezes Swin ``features`` layers ``[0, freeze_stages)``."""

    name: str = "default"
    seed: int = 946
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 20
    head_lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    num_workers: int = 8
    score_thresh: float = 0.05
    freeze_stages: int = 4
    min_size: int = 800
    max_size: int = 1333
    accum_steps: int = 2
    use_amp: bool = True
    train_eval_every: int = 5


def default_presets() -> list[TrainConfig]:
    """Named hyperparameter grids for ``--sweep`` or ``--preset``."""
    return [
        TrainConfig(
            name="default",
            head_lr=2e-4,
            backbone_lr=2e-5,
            freeze_stages=4,
            weight_decay=1e-4,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="head_lr_high",
            head_lr=5e-4,
            backbone_lr=1e-5,
            freeze_stages=4,
            weight_decay=1e-4,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="head_lr_low",
            head_lr=1e-4,
            backbone_lr=3e-5,
            freeze_stages=4,
            weight_decay=1e-4,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="freeze_more",
            head_lr=3e-4,
            backbone_lr=5e-5,
            freeze_stages=6,
            weight_decay=1e-4,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="freeze_less",
            head_lr=2e-4,
            backbone_lr=1e-5,
            freeze_stages=2,
            weight_decay=1e-4,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="wd_high",
            head_lr=2e-4,
            backbone_lr=2e-5,
            freeze_stages=4,
            weight_decay=5e-4,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="wd_low",
            head_lr=2e-4,
            backbone_lr=2e-5,
            freeze_stages=4,
            weight_decay=5e-5,
            warmup_epochs=5,
        ),
        TrainConfig(
            name="long_warmup",
            head_lr=2e-4,
            backbone_lr=2e-5,
            freeze_stages=4,
            weight_decay=1e-4,
            warmup_epochs=10,
        ),
    ]


def _select_device() -> torch.device:
    """Prefer CUDA, then MPS if usable, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


DEVICE = _select_device()


def set_seed(seed: int):
    """Fix RNGs for reproducibility (best-effort on GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _list_images(img_dir: Path) -> list[Path]:
    """Collect image paths for supported extensions, sorted by filename."""
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(img_dir.glob(f"*{ext}"))
    files = sorted(set(files), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No images ({IMAGE_EXTENSIONS}) under {img_dir}")
    return files


def _pil_can_decode(path: Path) -> bool:
    """Return True if PIL can fully decode ``path`` (filters corrupt files)."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def _filter_readable(paths: list[Path], label: str) -> list[Path]:
    """Keep only decodable images; log a short warning if any are dropped."""
    ok, bad = [], []
    for p in paths:
        (ok if _pil_can_decode(p) else bad).append(p)
    if bad:
        n = len(bad)
        sample = ", ".join(x.name for x in bad[:5])
        more = f" … (+{n - 5} more)" if n > 5 else ""
        print(f"  Warning [{label}]: skipped {n} corrupt image(s): {sample}{more}")
    return ok


class TrafficDetectionDataset(Dataset):
    """YOLO boxes → pixel xyxy + torchvision detection targets (``iscrowd``, ``area``)."""

    def __init__(self, data_root: Path, split: str, augment: bool = False):
        self.img_dir = Path(data_root) / "images" / split
        self.lbl_dir = Path(data_root) / "labels" / split
        raw = _list_images(self.img_dir)
        self.img_files = _filter_readable(raw, f"images/{split}")
        if not self.img_files:
            raise FileNotFoundError(
                f"No readable images under {self.img_dir} "
                f"(all {len(raw)} candidates failed)",
            )
        self.augment = augment
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02,
        )

    def __len__(self) -> int:
        return len(self.img_files)

    @staticmethod
    def _parse_yolo_label(path: Path, img_w: int, img_h: int):
        """Parse one label file; YOLO class ids are shifted by +1 for RetinaNet."""
        boxes, labels = [], []
        if not path.exists():
            return boxes, labels
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = (float(v) for v in parts[1:])
                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                x2 = (xc + bw / 2.0) * img_w
                y2 = (yc + bh / 2.0) * img_h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # foreground classes start at 1
        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = self._parse_yolo_label(lbl_path, w, h)

        boxes_t = (
            torch.as_tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_t = (
            torch.as_tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64)
        )

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                if boxes_t.shape[0] > 0:
                    boxes_t[:, [0, 2]] = w - boxes_t[:, [2, 0]]
            img = self.color_jitter(img)

        img_tensor = TF.to_tensor(img)

        if boxes_t.shape[0] > 0:
            boxes_t[:, 0].clamp_(min=0, max=w)
            boxes_t[:, 1].clamp_(min=0, max=h)
            boxes_t[:, 2].clamp_(min=0, max=w)
            boxes_t[:, 3].clamp_(min=0, max=h)
            keep = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])
            boxes_t = boxes_t[keep]
            labels_t = labels_t[keep]

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (
                (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
                if len(boxes_t) > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros(len(boxes_t), dtype=torch.int64),
        }
        return img_tensor, target


def _collate(batch):
    """Detection-style batching: list of images, list of target dicts."""
    return tuple(zip(*batch))


class SwinBackboneWithFPN(nn.Module):
    """Swin-T stages → multi-scale features; FPN outputs C4-style maps for RetinaNet."""

    BLOCK_STAGE_INDICES = (1, 3, 5, 7)

    def __init__(self, freeze_stages: int):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.stages = swin.features

        for i, stage in enumerate(self.stages):
            if i < freeze_stages:
                for param in stage.parameters():
                    param.requires_grad = False

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[96, 192, 384, 768],
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256

    def forward(self, x):
        fmaps = {}
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i in self.BLOCK_STAGE_INDICES:
                fmaps[str(len(fmaps))] = x.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
        return self.fpn(fmaps)


def build_model(cfg: TrainConfig) -> RetinaNet:
    """Construct RetinaNet with custom anchors and input resize bounds from ``cfg``."""
    backbone = SwinBackboneWithFPN(freeze_stages=cfg.freeze_stages)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    return RetinaNet(
        backbone,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_gen,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        score_thresh=cfg.score_thresh,
        nms_thresh=0.5,
        detections_per_img=300,
    )


def get_optimizer_and_scheduler(model: nn.Module, cfg: TrainConfig):
    """AdamW with separate LRs for ``backbone.stages`` vs FPN + detection head."""
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone.stages" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": other_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    def lr_lambda(epoch: int):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(cfg.warmup_epochs, 1)
        progress = (epoch - cfg.warmup_epochs) / max(
            cfg.num_epochs - cfg.warmup_epochs, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    accum_steps: int,
):
    """One pass; loss is scaled for gradient accumulation before backward."""
    model.train()
    running = 0.0
    n = len(loader)
    log_every = max(1, n // 10)
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / accum_steps
            scaler.scale(loss).backward()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / accum_steps
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        running += loss.item() * accum_steps
        if (i + 1) % log_every == 0 or (i + 1) == n:
            print(f"    batch [{i+1}/{n}]  loss={loss.item() * accum_steps:.4f}")

    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, scaler: GradScaler | None):
    """Mean RetinaNet training loss on ``loader`` (``model.train()`` required for loss dict)."""
    model.train()
    total = 0.0
    n = len(loader)
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if scaler is not None:
            with autocast(device_type="cuda"):
                loss_dict = model(images, targets)
        else:
            loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
    return total / max(n, 1)


@torch.no_grad()
def collect_predictions(model, loader, device, scaler: GradScaler | None):
    """Run NMS inference and return per-image dicts on CPU for metric code."""
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
        else:
            outputs = model(images)
        all_preds.extend({k: v.cpu() for k, v in o.items()} for o in outputs)
        all_targets.extend({k: v.cpu() for k, v in t.items()} for t in targets)
    return all_preds, all_targets


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the precision–recall curve (VOC-style integral)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _evaluate_at_iou(all_preds, all_targets, iou_thresh: float):
    """Greedy per-image matching; returns per-class AP and PR arrays at one IoU."""
    cls_data = defaultdict(lambda: {"scores": [], "tp": [], "n_gt": 0})

    for pred, tgt in zip(all_preds, all_targets):
        gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]
        pd_boxes, pd_scores, pd_labels = (
            pred["boxes"], pred["scores"], pred["labels"],
        )

        for lbl in gt_labels.tolist():
            cls_data[lbl]["n_gt"] += 1

        matched: set = set()
        for pidx in pd_scores.argsort(descending=True):
            cls = pd_labels[pidx].item()
            cls_data[cls]["scores"].append(pd_scores[pidx].item())

            best_iou, best_gi = 0.0, -1
            mask = gt_labels == cls
            if mask.any():
                cls_gt_boxes = gt_boxes[mask]
                cls_gt_idx = mask.nonzero(as_tuple=False).squeeze(1)
                ious = box_iou(
                    pd_boxes[pidx].unsqueeze(0), cls_gt_boxes
                ).squeeze(0)
                for j, gi in enumerate(cls_gt_idx.tolist()):
                    iou_val = ious[j].item()
                    if iou_val > best_iou and gi not in matched:
                        best_iou, best_gi = iou_val, gi

            if best_iou >= iou_thresh and best_gi >= 0:
                cls_data[cls]["tp"].append(1)
                matched.add(best_gi)
            else:
                cls_data[cls]["tp"].append(0)

    results: dict = {}
    pr_curves: dict = {}

    for cls_id in sorted(cls_data):
        d = cls_data[cls_id]
        if d["n_gt"] == 0:
            continue
        scores = np.array(d["scores"])
        tp = np.array(d["tp"])
        order = np.argsort(-scores)
        tp = tp[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        recall = tp_cum / d["n_gt"]
        precision = tp_cum / (tp_cum + fp_cum)

        ap = _compute_ap(recall, precision)
        f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
        max_f1 = float(f1_vals.max()) if len(f1_vals) > 0 else 0.0

        name = CLASS_NAMES.get(cls_id - 1, f"class_{cls_id}")
        results[name] = {
            "precision": float(precision[-1]) if len(precision) else 0.0,
            "recall": float(recall[-1]) if len(recall) else 0.0,
            "AP": ap,
            "max_f1": max_f1,
        }
        pr_curves[name] = (recall.copy(), precision.copy())

    return results, pr_curves


def evaluate(all_preds, all_targets):
    """mAP@0.5, mAP@0.5:0.95 (mean over IoU 0.5:0.95 step 0.05), F1 at IoU 0.5."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_per_thresh = []
    results_50, pr_curves_50 = None, None

    for iou_t in iou_thresholds:
        results, pr_curves = _evaluate_at_iou(all_preds, all_targets, float(iou_t))
        mean_ap = (
            float(np.mean([v["AP"] for v in results.values()])) if results else 0.0
        )
        aps_per_thresh.append(mean_ap)
        if abs(iou_t - 0.5) < 1e-6:
            results_50 = results
            pr_curves_50 = pr_curves

    mAP_50 = aps_per_thresh[0] if aps_per_thresh else 0.0
    mAP_50_95 = float(np.mean(aps_per_thresh)) if aps_per_thresh else 0.0
    mean_f1 = (
        float(np.mean([v["max_f1"] for v in results_50.values()]))
        if results_50 else 0.0
    )
    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "mean_f1": mean_f1,
        "per_class": results_50 or {},
        "pr_curves": pr_curves_50 or {},
    }


def _save_dual_curve(
    train_vals: list[float],
    val_vals: list[float],
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    ylim: tuple[float, float] | None = None,
):
    """Plot train vs val series; train may be shorter (interpolated x-axis)."""
    if not val_vals:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ep_val = range(1, len(val_vals) + 1)
    ax.plot(ep_val, val_vals, "s-", markersize=3, label="Validation")
    if train_vals:
        ep_tr = np.linspace(1, len(val_vals), len(train_vals))
        ax.plot(ep_tr, train_vals, "o-", markersize=3, label="Train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        all_vals = list(val_vals) + list(train_vals or [])
        if all_vals and max(all_vals) <= 1.05:
            ax.set_ylim(0, 1.0)
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_loss_curves(train_losses, val_losses, out_dir: Path):
    """Write ``loss_curves.png``."""
    _save_dual_curve(
        train_losses, val_losses, "Loss",
        "Swin-T + RetinaNet — Training & Validation Loss",
        "loss_curves.png", out_dir,
    )


def save_ap50_curves(train_vals, val_vals, out_dir: Path):
    """Write ``ap50_curves.png``."""
    _save_dual_curve(
        train_vals, val_vals, "mAP @ IoU=0.5",
        "Swin-T + RetinaNet — mAP@0.5 (Train vs Val)",
        "ap50_curves.png", out_dir,
    )


def save_ap50_95_curves(train_vals, val_vals, out_dir: Path):
    """Write ``ap50_95_curves.png``."""
    _save_dual_curve(
        train_vals, val_vals, "mAP @ IoU=0.50:0.95",
        "Swin-T + RetinaNet — mAP@0.50:0.95 (Train vs Val)",
        "ap50_95_curves.png", out_dir,
    )


def save_f1_curves(train_vals, val_vals, out_dir: Path):
    """Write ``f1_curves.png``."""
    _save_dual_curve(
        train_vals, val_vals, "F1 Score (max from PR curve)",
        "Swin-T + RetinaNet — F1 Score (Train vs Val)",
        "f1_curves.png", out_dir,
    )


def save_lr_curves(head_lrs: list[float], backbone_lrs: list[float], out_dir: Path):
    """Plot both optimizer group learning rates (log scale)."""
    if not head_lrs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(head_lrs) + 1)
    ax.plot(epochs, head_lrs, label="Head LR", linewidth=2)
    ax.plot(epochs, backbone_lrs, label="Backbone LR", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "lr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_train_val_gap_ap50(train_ap50, val_ap50, out_dir: Path):
    """Plot train minus val AP@0.5 per epoch (overfitting when positive)."""
    if not train_ap50 or not val_ap50:
        return
    n = min(len(train_ap50), len(val_ap50))
    gap = [train_ap50[i] - val_ap50[i] for i in range(n)]
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, n + 1)
    ax.plot(epochs, gap, "o-", markersize=3, color="tab:purple",
            label="Train − Val AP@0.5")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap (overfit if positive)")
    ax.set_title("Overfitting diagnostic — Train vs Val AP@0.5 gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "train_val_gap_ap50.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_per_class_metric_curves(
    per_class: dict[str, list[float]], ylabel: str,
    title: str, fname: str, out_dir: Path,
):
    """Per-class validation metric vs epoch (e.g. AP@0.5 or F1)."""
    if not per_class:
        return
    n = max(len(v) for v in per_class.values()) if per_class else 0
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in per_class.items():
        if len(vals) == n:
            ax.plot(range(1, n + 1), vals, "o-", markersize=2, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_metrics_overview(
    train_losses, val_losses, train_ap50, val_ap50,
    train_ap5095, val_ap5095, train_f1, val_f1, out_dir: Path,
):
    """2×2 summary figure: loss, mAP@0.5, mAP@0.5:0.95, mean F1."""
    if not val_losses:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ep = range(1, len(val_losses) + 1)

    axes[0, 0].plot(ep, train_losses, label="Train")
    axes[0, 0].plot(ep, val_losses, label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, val_ap50, label="Val")
    if train_ap50:
        ep_tr = np.linspace(1, len(val_ap50), len(train_ap50))
        axes[0, 1].plot(ep_tr, train_ap50, label="Train")
    axes[0, 1].set_title("mAP @ 0.5")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ep, val_ap5095, label="Val")
    if train_ap5095:
        ep_tr = np.linspace(1, len(val_ap5095), len(train_ap5095))
        axes[1, 0].plot(ep_tr, train_ap5095, label="Train")
    axes[1, 0].set_title("mAP @ 0.50:0.95")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep, val_f1, label="Val")
    if train_f1:
        ep_tr = np.linspace(1, len(val_f1), len(train_f1))
        axes[1, 1].plot(ep_tr, train_f1, label="Train")
    axes[1, 1].set_title("Mean F1")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Training overview — Swin-T + RetinaNet", fontsize=12)
    fig.tight_layout()
    path = out_dir / "metrics_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_precision_recall_bars(per_class: dict, out_dir: Path, split_name: str):
    """Grouped bar chart of last-threshold precision and recall per class."""
    names = list(per_class.keys())
    if not names:
        return
    prec = [per_class[n]["precision"] for n in names]
    rec = [per_class[n]["recall"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, prec, w, label="Precision", color="tab:blue")
    ax.bar(x + w / 2, rec, w, label="Recall", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision & Recall per class ({split_name}, IoU ≥ 0.5)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    tag = split_name.lower().replace(" ", "_")
    path = out_dir / f"precision_recall_bars_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_ap_per_iou_bar(aps: list[float], out_dir: Path):
    """Bar chart of mean AP vs IoU on the test split."""
    if not aps:
        return
    labels = [f"{0.5 + 0.05 * i:.2f}" for i in range(len(aps))]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, aps, color="tab:green", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Mean AP")
    ax.set_title("Mean AP vs IoU threshold (test set)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "ap_per_iou_threshold_test.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def save_pr_curves(pr_curves: dict, out_dir: Path):
    """Per-class precision–recall curves at IoU 0.5."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls_name, (rec, prec) in pr_curves.items():
        ax.plot(rec, prec, linewidth=2, label=cls_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (IoU ≥ 0.5, test)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def compute_ap_per_iou_threshold(all_preds, all_targets) -> list[float]:
    """Mean AP at each COCO IoU threshold (for the test-set bar chart)."""
    out = []
    for iou_t in np.arange(0.5, 1.0, 0.05):
        results, _ = _evaluate_at_iou(all_preds, all_targets, float(iou_t))
        m = float(np.mean([v["AP"] for v in results.values()])) if results else 0.0
        out.append(m)
    return out


def run_training(
    cfg: TrainConfig,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Full fit: train/val loop, plots, best checkpoint by val mAP@0.5, test metrics."""
    set_seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Device : {DEVICE}")
    print(f"Data   : {data_dir}")
    print(f"Output : {output_dir}")
    print(f"Config : {cfg.name}  head_lr={cfg.head_lr}  backbone_lr={cfg.backbone_lr}  "
          f"freeze_stages={cfg.freeze_stages}  wd={cfg.weight_decay}")
    print(f"  batch={cfg.batch_size}  eval_batch={cfg.eval_batch_size}  "
          f"accum={cfg.accum_steps}  (effective batch={cfg.batch_size * cfg.accum_steps})  "
          f"AMP={cfg.use_amp}  train_eval_every={cfg.train_eval_every}")

    train_ds = TrafficDetectionDataset(data_dir, "train", augment=True)
    val_ds = TrafficDetectionDataset(data_dir, "val", augment=False)
    test_ds = TrafficDetectionDataset(data_dir, "test", augment=False)
    train_eval_ds = TrafficDetectionDataset(data_dir, "train", augment=False)

    print(f"\n=== Dataset sizes ===")
    print(f"  train : {len(train_ds)}")
    print(f"  val   : {len(val_ds)}")
    print(f"  test  : {len(test_ds)}")

    pin = DEVICE.type == "cuda"
    pw = cfg.num_workers > 0
    base_kw = dict(
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=pin,
        persistent_workers=pw,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **base_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, **base_kw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.eval_batch_size, shuffle=False, **base_kw,
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=cfg.eval_batch_size, shuffle=False, **base_kw,
    )

    model = build_model(cfg)
    model.to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n  Parameters : {n_total:.1f} M total, {n_train:.1f} M trainable")
    print(f"  Frozen     : stages 0–{cfg.freeze_stages - 1}  |  "
          f"Fine-tuned : stages {cfg.freeze_stages}–7")

    optimizer, scheduler = get_optimizer_and_scheduler(model, cfg)
    scaler = GradScaler("cuda") if (cfg.use_amp and DEVICE.type == "cuda") else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ap50: list[float] = []
    val_ap50_95: list[float] = []
    val_f1: list[float] = []
    train_ap50: list[float] = []
    train_ap50_95: list[float] = []
    train_f1: list[float] = []
    head_lrs: list[float] = []
    backbone_lrs: list[float] = []

    cls_names = list(CLASS_NAMES.values())
    val_ap50_per_class: dict[str, list[float]] = {c: [] for c in cls_names}
    val_f1_per_class: dict[str, list[float]] = {c: [] for c in cls_names}

    best_val_map = 0.0

    print(f"\n=== Training {cfg.num_epochs} epochs ===\n")

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        ep1 = epoch + 1

        trn_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, scaler, cfg.accum_steps,
        )
        val_loss = validate(model, val_loader, DEVICE, scaler)

        val_preds, val_targets = collect_predictions(model, val_loader, DEVICE, scaler)
        val_m = evaluate(val_preds, val_targets)

        do_train_eval = (
            ep1 % cfg.train_eval_every == 0
            or ep1 == 1
            or ep1 == cfg.num_epochs
        )
        if do_train_eval:
            trn_preds, trn_targets = collect_predictions(
                model, train_eval_loader, DEVICE, scaler,
            )
            trn_m = evaluate(trn_preds, trn_targets)
            train_ap50.append(trn_m["mAP_50"])
            train_ap50_95.append(trn_m["mAP_50_95"])
            train_f1.append(trn_m["mean_f1"])

        scheduler.step()
        head_lrs.append(optimizer.param_groups[1]["lr"])
        backbone_lrs.append(optimizer.param_groups[0]["lr"])

        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        val_ap50.append(val_m["mAP_50"])
        val_ap50_95.append(val_m["mAP_50_95"])
        val_f1.append(val_m["mean_f1"])

        for c in cls_names:
            pc = val_m["per_class"].get(c)
            val_ap50_per_class[c].append(pc["AP"] if pc else 0.0)
            val_f1_per_class[c].append(pc["max_f1"] if pc else 0.0)

        lr_h = head_lrs[-1]
        elapsed = time.time() - t0
        trn_tag = (
            f"AP50 {trn_m['mAP_50']:.4f}/"
            if do_train_eval
            else "AP50 --.--/"
        )
        ap95_tag = (
            f"AP5095 {trn_m['mAP_50_95']:.4f}/"
            if do_train_eval
            else "AP5095 --.--/"
        )
        f1_tag = (
            f"F1 {trn_m['mean_f1']:.4f}/"
            if do_train_eval
            else "F1 --.--/"
        )
        print(
            f"  Epoch {ep1:>3}/{cfg.num_epochs}  "
            f"loss {trn_loss:.4f}/{val_loss:.4f}  "
            f"{trn_tag}{val_m['mAP_50']:.4f}  "
            f"{ap95_tag}{val_m['mAP_50_95']:.4f}  "
            f"{f1_tag}{val_m['mean_f1']:.4f}  "
            f"lr {lr_h:.2e}  ({elapsed:.0f}s)"
        )

        if val_m["mAP_50"] > best_val_map:
            best_val_map = val_m["mAP_50"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"    ✓ best_model.pt (val mAP@0.5={best_val_map:.4f})")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print("\nTraining complete.  Models saved to:", output_dir)

    save_loss_curves(train_losses, val_losses, output_dir)
    save_ap50_curves(train_ap50, val_ap50, output_dir)
    save_ap50_95_curves(train_ap50_95, val_ap50_95, output_dir)
    save_f1_curves(train_f1, val_f1, output_dir)
    save_lr_curves(head_lrs, backbone_lrs, output_dir)
    save_train_val_gap_ap50(train_ap50, val_ap50, output_dir)
    save_metrics_overview(
        train_losses, val_losses,
        train_ap50, val_ap50,
        train_ap50_95, val_ap50_95,
        train_f1, val_f1,
        output_dir,
    )
    save_per_class_metric_curves(
        val_ap50_per_class, "AP @ IoU=0.5",
        "Per-class AP@0.5 on validation (by epoch)",
        "per_class_ap50_val.png", output_dir,
    )
    save_per_class_metric_curves(
        val_f1_per_class, "F1",
        "Per-class F1 on validation (by epoch)",
        "per_class_f1_val.png", output_dir,
    )

    print("\n=== Test evaluation (best checkpoint) ===")
    state = torch.load(
        output_dir / "best_model.pt", map_location=DEVICE, weights_only=True,
    )
    model.load_state_dict(state)
    model.to(DEVICE)

    preds, targets = collect_predictions(model, test_loader, DEVICE, scaler)
    test_m = evaluate(preds, targets)

    print(f"\n{'=' * 70}")
    print(f"  TEST — preset {cfg.name}")
    print(f"{'=' * 70}")
    for name, m in test_m["per_class"].items():
        print(
            f"  {name:15s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"AP@0.5={m['AP']:.4f}  F1={m['max_f1']:.4f}"
        )
    print(f"  {'mAP@0.5':15s}  {test_m['mAP_50']:.4f}")
    print(f"  {'mAP@0.50:0.95':15s}  {test_m['mAP_50_95']:.4f}")
    print(f"  {'mean F1':15s}  {test_m['mean_f1']:.4f}")
    print(f"{'=' * 70}\n")

    save_pr_curves(test_m["pr_curves"], output_dir)
    save_precision_recall_bars(test_m["per_class"], output_dir, "test")

    aps_per_iou = compute_ap_per_iou_threshold(preds, targets)
    save_ap_per_iou_bar(aps_per_iou, output_dir)

    summary = {
        "preset": cfg.name,
        "best_val_mAP_50": best_val_map,
        "test_mAP_50": test_m["mAP_50"],
        "test_mAP_50_95": test_m["mAP_50_95"],
        "test_mean_f1": test_m["mean_f1"],
        "per_class_test": {
            k: {kk: v[kk] for kk in ("AP", "precision", "recall", "max_f1")}
            for k, v in test_m["per_class"].items()
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    """Define CLI; defaults for paths come from module-level ``DEFAULT_*``."""
    p = argparse.ArgumentParser(
        description="Swin-T + RetinaNet fine-tuning on YOLO data."
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--backbone-lr", type=float, default=None)
    p.add_argument("--freeze-stages", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--train-eval-every", type=int, default=None,
                   help="Evaluate on train set every N epochs (default 5)")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def _merge_cli(cfg: TrainConfig, args) -> TrainConfig:
    """Overlay non-None CLI arguments onto ``cfg``."""
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        cfg.eval_batch_size = args.eval_batch_size
    if args.head_lr is not None:
        cfg.head_lr = args.head_lr
    if args.backbone_lr is not None:
        cfg.backbone_lr = args.backbone_lr
    if args.freeze_stages is not None:
        cfg.freeze_stages = args.freeze_stages
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed
    if args.accum_steps is not None:
        cfg.accum_steps = args.accum_steps
    if args.no_amp:
        cfg.use_amp = False
    if args.train_eval_every is not None:
        cfg.train_eval_every = args.train_eval_every
    return cfg


def main():
    """Entry: single run, or ``--sweep`` over presets into subfolders."""
    args = parse_args()
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    base_out: Path = args.output_dir

    presets = {c.name: c for c in default_presets()}
    if args.preset is not None and args.preset not in presets:
        raise SystemExit(
            f"Unknown --preset {args.preset!r}. Choose from: {sorted(presets)}"
        )

    if args.sweep:
        sweep_list = [presets[args.preset]] if args.preset else list(default_presets())
        sweep_results: list[dict[str, Any]] = []
        for cfg in sweep_list:
            cfg = _merge_cli(cfg, args)
            out = base_out / cfg.name
            print(f"\n########## Sweep: {cfg.name} → {out} ##########\n")
            summary = run_training(cfg, data_dir, out)
            sweep_results.append(summary)
        with open(base_out / "sweep_summary.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep summary written to {base_out / 'sweep_summary.json'}")
        return

    cfg = presets[args.preset] if args.preset else TrainConfig(name="single")
    cfg = _merge_cli(cfg, args)
    out_dir = base_out / args.run_name if args.run_name else base_out
    run_training(cfg, data_dir, out_dir)


if __name__ == "__main__":
    main()
