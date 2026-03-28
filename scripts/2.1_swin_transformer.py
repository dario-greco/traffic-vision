#!/usr/bin/env python3
"""
2.1_swin_transformer.py
========================
Swin-T (Tiny) backbone + Feature Pyramid Network + RetinaNet head
for detecting **stop signs** (class 0) and **traffic lights** (class 1).

Dataset layout  (YOLO format):
    yolo_dataset/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt   →  cls  x_c  y_c  w  h  (normalised)

Usage:
    python scripts/2.1_swin_transformer.py
"""

import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "yolo_dataset"
OUTPUT_DIR = BASE_DIR / "swin_runs"

CLASS_NAMES = {0: "stop sign", 1: "traffic light"}
NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 for background 

SEED = 946
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 5
LR_GAMMA = 0.5
IOU_THRESHOLD = 0.5
NUM_WORKERS = 0  


def _select_device() -> torch.device:
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

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Dataset 
# ============================================================
class TrafficDetectionDataset(Dataset):
    def __init__(self, split: str):
        self.img_dir = DATA_DIR / "images" / split
        self.lbl_dir = DATA_DIR / "labels" / split
        self.img_files = sorted(self.img_dir.glob("*.jpg"))
        if not self.img_files:
            raise FileNotFoundError(f"No .jpg images in {self.img_dir}")
        self._to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.img_files)

    def _parse_yolo_label(self, path: Path, img_w: int, img_h: int):
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
                labels.append(cls + 1)  
        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        lbl_path = self.lbl_dir / img_path.with_suffix(".txt").name

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = self._parse_yolo_label(lbl_path, w, h)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if len(boxes_t) > 0 else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes_t), dtype=torch.int64),
        }
        return self._to_tensor(img), target


def _collate(batch):
    return tuple(zip(*batch))

# ============================================================
# Model — Swin-T backbone + FPN + RetinaNet
# ============================================================
class SwinBackboneWithFPN(nn.Module):
    BLOCK_STAGE_INDICES = (1, 3, 5, 7)

    def __init__(self):
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.stages = swin.features  

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[96, 192, 384, 768],
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256  

    def forward(self, x):
        feature_maps = {}
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i in self.BLOCK_STAGE_INDICES:
                feature_maps[str(len(feature_maps))] = x.permute(0, 3, 1, 2).contiguous()
        return self.fpn(feature_maps)


def build_model() -> RetinaNet:
    backbone = SwinBackboneWithFPN()

    for param in backbone.stages.parameters():
        param.requires_grad = False

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = RetinaNet(
        backbone,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_gen,
        min_size=640,
        max_size=800,
    )
    return model

# ============================================================
# Training helpers
# ============================================================
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running = 0.0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += loss.item()
        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            print(f"    batch [{i+1}/{len(loader)}]  loss={loss.item():.4f}")

    return running / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.train() # RetinaNet needs to be in train mode to return losses
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
    return total / max(len(loader), 1)

# ============================================================
# Evaluation 
# ============================================================
@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        all_preds.extend({k: v.cpu() for k, v in o.items()} for o in outputs)
        all_targets.extend({k: v.cpu() for k, v in t.items()} for t in targets)
    return all_preds, all_targets

def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))

def evaluate(all_preds, all_targets, iou_thresh: float = IOU_THRESHOLD):
    cls_data = defaultdict(lambda: {"scores": [], "tp": [], "n_gt": 0})

    for pred, tgt in zip(all_preds, all_targets):
        gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]
        pd_boxes, pd_scores, pd_labels = (
            pred["boxes"],
            pred["scores"],
            pred["labels"],
        )

        for lbl in gt_labels.tolist():
            cls_data[lbl]["n_gt"] += 1

        matched: set = set()
        for idx in pd_scores.argsort(descending=True):
            cls = pd_labels[idx].item()
            cls_data[cls]["scores"].append(pd_scores[idx].item())

            best_iou, best_gi = 0.0, -1
            mask = gt_labels == cls
            if mask.any():
                cls_gt_boxes = gt_boxes[mask]
                cls_gt_idx = mask.nonzero(as_tuple=False).squeeze(1)
                ious = box_iou(pd_boxes[idx].unsqueeze(0), cls_gt_boxes).squeeze(0)
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
        name = CLASS_NAMES.get(cls_id - 1, f"class_{cls_id}")
        results[name] = {
            "precision": float(precision[-1]),
            "recall": float(recall[-1]),
            "AP": ap,
        }
        pr_curves[name] = (recall.copy(), precision.copy())

    mAP = float(np.mean([v["AP"] for v in results.values()])) if results else 0.0
    return results, mAP, pr_curves

# ============================================================
# Plotting
# ============================================================
def save_loss_curves(train_losses, val_losses, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "o-", label="Train")
    ax.plot(epochs, val_losses, "s-", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Swin-T + RetinaNet — Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

def save_pr_curves(pr_curves: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls_name, (rec, prec) in pr_curves.items():
        ax.plot(rec, prec, linewidth=2, label=cls_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (IoU ≥ 0.5)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

# ============================================================
# Main entry point
# ============================================================
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Data   : {DATA_DIR}")
    print(f"Output : {OUTPUT_DIR}")

    print("\n=== Loading datasets ===")
    train_ds = TrafficDetectionDataset("train")
    val_ds = TrafficDetectionDataset("val")  
    test_ds = TrafficDetectionDataset("test")
    print(f"  train : {len(train_ds)} images")
    print(f"  val   : {len(val_ds)} images")
    print(f"  test  : {len(test_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=_collate,
    )

    print("\n=== Building Swin-T + RetinaNet ===")
    model = build_model()
    model.to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters : {n_total:.1f} M total, {n_train:.1f} M trainable (Backbone Frozen)")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")

    print(f"\n=== Training for {NUM_EPOCHS} epochs ===\n")
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        trn_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()

        train_losses.append(trn_loss)
        val_losses.append(val_loss)

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch + 1:>2}/{NUM_EPOCHS}  "
            f"train_loss={trn_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={lr_now:.1e}  ({elapsed:.0f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"    ✓ Best model saved (val_loss={val_loss:.4f})")
        print()

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    print("Training complete.  Models saved to:", OUTPUT_DIR)
    save_loss_curves(train_losses, val_losses, OUTPUT_DIR)

    print("\n=== Final evaluation on TEST set ===")
    state = torch.load(
        OUTPUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=True,
    )
    model.load_state_dict(state)
    model.to(DEVICE)

    preds, targets = collect_predictions(model, test_loader, DEVICE)
    results, mAP, pr_curves = evaluate(preds, targets)

    print(f"\n{'=' * 60}")
    print(f"  TEST RESULTS  (IoU ≥ {IOU_THRESHOLD})")
    print(f"{'=' * 60}")
    for name, m in results.items():
        print(
            f"  {name:15s}  Precision={m['precision']:.4f}  "
            f"Recall={m['recall']:.4f}  AP={m['AP']:.4f}"
        )
    print(f"  {'mAP':15s}  {mAP:.4f}")
    print(f"{'=' * 60}\n")

    save_pr_curves(pr_curves, OUTPUT_DIR)
    print("\nAll outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()