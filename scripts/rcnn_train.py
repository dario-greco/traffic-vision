"""
Train Faster R-CNN on the data_final YOLO-format dataset.
Modularized for Hyperparameter Optimization (HPO).
"""

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF

# --- Configuration & Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "data_final")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "detect", "rcnn_runs")
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

@dataclass
class DatasetStats:
    total_images: int = 0
    missing_label_files: int = 0
    dropped_bad_lines: int = 0
    dropped_bad_boxes: int = 0

class DataFinalDetectionDataset(Dataset):
    """Reads YOLO txt labels and returns tensors compatible with torchvision detection."""
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.stats = DatasetStats()

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Missing images directory: {images_dir}")
        self.image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(SUPPORTED_EXTS))
        self.stats.total_images = len(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_dir, image_name)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_tensor = TF.to_tensor(image)
        h, w = image_tensor.shape[1], image_tensor.shape[2]

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    class_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:])
                    x1 = max(0.0, (x_c - bw / 2.0) * w)
                    y1 = max(0.0, (y_c - bh / 2.0) * h)
                    x2 = min(float(w), (x_c + bw / 2.0) * w)
                    y2 = min(float(h), (y_c + bh / 2.0) * h)
                    if (x2 - x1) > 1e-3 and (y2 - y1) > 1e-3:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id + 1)
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index]),
            "area": torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]) if boxes else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        return image_tensor, target

def get_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))

def _compute_ap_101(recalls, precisions):
    if recalls.numel() == 0: return 0.0
    for i in range(precisions.numel() - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    recall_grid = torch.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_grid:
        idx = torch.where(recalls >= r)[0]
        ap += float(precisions[idx[0]]) if idx.numel() > 0 else 0.0
    return ap / 101.0

def evaluate_map(model, loader, device, num_classes: int):
    model.eval()
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    class_ids = list(range(1, num_classes))
    gt_by_class = {c: {} for c in class_ids}
    pred_by_class = {c: [] for c in class_ids}

    with torch.no_grad():
        img_idx = 0
        for images, targets in loader:
            outputs = model([img.to(device) for img in images])
            for output, target in zip(outputs, targets):
                for cls in class_ids:
                    gt_by_class[cls][img_idx] = target["boxes"][target["labels"] == cls]
                    mask = output["labels"] == cls
                    for b, s in zip(output["boxes"][mask], output["scores"][mask]):
                        pred_by_class[cls].append({"img_id": img_idx, "box": b.cpu(), "score": s.cpu()})
                img_idx += 1

    aps = []
    for thr in iou_thresholds:
        thr_aps = []
        for cls in class_ids:
            preds = sorted(pred_by_class[cls], key=lambda x: x["score"], reverse=True)
            num_gt = sum(len(g) for g in gt_by_class[cls].values())
            if num_gt == 0: continue
            tp, fp = torch.zeros(len(preds)), torch.zeros(len(preds))
            matched = {k: torch.zeros(len(v), dtype=torch.bool) for k, v in gt_by_class[cls].items()}
            for i, p in enumerate(preds):
                gt_boxes = gt_by_class[cls][p["img_id"]]
                if len(gt_boxes) > 0:
                    ious = box_iou(p["box"].unsqueeze(0), gt_boxes).squeeze(0)
                    m_iou, m_idx = torch.max(ious, 0)
                    if m_iou >= thr and not matched[p["img_id"]][m_idx]:
                        tp[i], matched[p["img_id"]][m_idx] = 1.0, True
                    else: fp[i] = 1.0
                else: fp[i] = 1.0
            c_tp, c_fp = torch.cumsum(tp, 0), torch.cumsum(fp, 0)
            rec, prec = c_tp / num_gt, c_tp / torch.clamp(c_tp + c_fp, min=1e-8)
            thr_aps.append(_compute_ap_101(rec, prec))
        if thr_aps: aps.append(sum(thr_aps)/len(thr_aps))
    
    return {
        "mAP_50_95": sum(aps)/len(aps) if aps else 0.0,
        "MAP_50": aps[0] if aps else 0.0,
        "MAP_75": aps[5] if len(aps) > 5 else 0.0
    }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-name", default="baseline_rcnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_known_args()[0]

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_img = os.path.join(args.dataset_dir, "images", "train")
    train_lab = os.path.join(args.dataset_dir, "labels", "train")
    val_img = os.path.join(args.dataset_dir, "images", "val")
    val_lab = os.path.join(args.dataset_dir, "labels", "val")

    loader = DataLoader(DataFinalDetectionDataset(train_img, train_lab), 
                        batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.num_workers, collate_fn=collate_fn)
    
    val_loader = DataLoader(DataFinalDetectionDataset(val_img, val_lab), 
                            batch_size=1, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = get_model(num_classes=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    losses, best_map, patience_counter = [], -1.0, 0

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()
        losses.append(avg_loss)
        
        metrics = evaluate_map(model, val_loader, device, num_classes=3)
        cur_map = metrics["mAP_50_95"]
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | mAP: {cur_map:.4f}")

        if cur_map > best_map + 0.001:
            best_map = cur_map
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "faster_rcnn_best.pt"))
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    results = {**metrics, "final_loss": losses[-1], "best_loss": min(losses), "epochs_completed": len(losses)}
    with open(os.path.join(run_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    train_model(get_args())