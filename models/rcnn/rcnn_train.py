"""Train Faster R-CNN on the data_final YOLO-format dataset."""

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
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

        self.image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(SUPPORTED_EXTS)
        )
        self.stats.total_images = len(self.image_files)

        if not self.image_files:
            raise RuntimeError(f"No supported images found in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_dir, image_name)

        # PIL is the most portable decoder across cluster environments.
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_tensor = TF.to_tensor(image)

        h, w = image_tensor.shape[1], image_tensor.shape[2]

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        self.stats.dropped_bad_lines += 1
                        continue

                    try:
                        class_id = int(parts[0])
                        x_center, y_center, bw, bh = map(float, parts[1:])
                    except ValueError:
                        self.stats.dropped_bad_lines += 1
                        continue

                    # Filter invalid numeric values and non-positive box sizes.
                    vals = [x_center, y_center, bw, bh]
                    if any((not math.isfinite(v)) for v in vals) or bw <= 0.0 or bh <= 0.0:
                        self.stats.dropped_bad_boxes += 1
                        continue

                    x1 = max(0.0, (x_center - bw / 2.0) * w)
                    y1 = max(0.0, (y_center - bh / 2.0) * h)
                    x2 = min(float(w), (x_center + bw / 2.0) * w)
                    y2 = min(float(h), (y_center + bh / 2.0) * h)

                    if (x2 - x1) <= 1e-3 or (y2 - y1) <= 1e-3:
                        self.stats.dropped_bad_boxes += 1
                        continue

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # Background is class 0 in Faster R-CNN.
        else:
            self.stats.missing_label_files += 1

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
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

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {i}: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 0:
            print(f"Epoch {epoch} | Step {i}/{len(loader)} | Avg loss: {running_loss / i:.4f}")

    return running_loss / max(1, len(loader))


def _compute_ap_101(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """COCO-style AP using 101-point interpolation."""
    if recalls.numel() == 0 or precisions.numel() == 0:
        return 0.0

    # Enforce monotonically decreasing precision envelope.
    for i in range(precisions.numel() - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_grid = torch.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_grid:
        idx = torch.where(recalls >= r)[0]
        p = precisions[idx[0]] if idx.numel() > 0 else torch.tensor(0.0)
        ap += float(p)
    return ap / 101.0


def evaluate_map(model, loader, device, num_classes: int) -> Dict[str, float]:
    """Compute mAP_50_95, MAP_50, MAP_75 on a validation loader."""
    model.eval()

    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    class_ids = list(range(1, num_classes))  # exclude background

    gt_by_class: Dict[int, Dict[int, torch.Tensor]] = {c: {} for c in class_ids}
    pred_by_class: Dict[int, List[Dict[str, torch.Tensor]]] = {c: [] for c in class_ids}

    with torch.no_grad():
        image_index = 0
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                pred_boxes = output["boxes"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()

                for cls in class_ids:
                    gt_mask = gt_labels == cls
                    cls_gt_boxes = gt_boxes[gt_mask]
                    gt_by_class[cls][image_index] = cls_gt_boxes

                    pred_mask = pred_labels == cls
                    cls_pred_boxes = pred_boxes[pred_mask]
                    cls_pred_scores = pred_scores[pred_mask]
                    for b, s in zip(cls_pred_boxes, cls_pred_scores):
                        pred_by_class[cls].append(
                            {
                                "image_id": torch.tensor(image_index),
                                "box": b,
                                "score": s,
                            }
                        )

                image_index += 1

    aps_per_threshold: Dict[float, List[float]] = {thr: [] for thr in iou_thresholds}

    for cls in class_ids:
        gt_per_image = gt_by_class[cls]
        preds = sorted(pred_by_class[cls], key=lambda x: float(x["score"]), reverse=True)
        num_gt = int(sum(gt.shape[0] for gt in gt_per_image.values()))

        if num_gt == 0:
            continue

        for thr in iou_thresholds:
            matched = {img_id: torch.zeros(gt.shape[0], dtype=torch.bool) for img_id, gt in gt_per_image.items()}

            tp = torch.zeros(len(preds), dtype=torch.float32)
            fp = torch.zeros(len(preds), dtype=torch.float32)

            for i, pred in enumerate(preds):
                img_id = int(pred["image_id"])
                pbox = pred["box"].unsqueeze(0)
                gt_boxes = gt_per_image.get(img_id, torch.zeros((0, 4), dtype=torch.float32))

                if gt_boxes.shape[0] == 0:
                    fp[i] = 1.0
                    continue

                ious = box_iou(pbox, gt_boxes).squeeze(0)
                max_iou, max_idx = torch.max(ious, dim=0)

                if max_iou >= thr and not matched[img_id][max_idx]:
                    tp[i] = 1.0
                    matched[img_id][max_idx] = True
                else:
                    fp[i] = 1.0

            cum_tp = torch.cumsum(tp, dim=0)
            cum_fp = torch.cumsum(fp, dim=0)
            recalls = cum_tp / max(float(num_gt), 1.0)
            precisions = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-8)

            ap = _compute_ap_101(recalls, precisions)
            aps_per_threshold[thr].append(ap)

    ap50 = float(sum(aps_per_threshold[0.5]) / len(aps_per_threshold[0.5])) if aps_per_threshold[0.5] else 0.0
    ap75 = float(sum(aps_per_threshold[0.75]) / len(aps_per_threshold[0.75])) if aps_per_threshold[0.75] else 0.0

    all_aps = [ap for thr in iou_thresholds for ap in aps_per_threshold[thr]]
    map_50_95 = float(sum(all_aps) / len(all_aps)) if all_aps else 0.0

    return {
        "mAP_50_95": map_50_95,
        "MAP_50": ap50,
        "MAP_75": ap75,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on data_final")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-name", default="traffic_model_rcnn_data_final")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = os.path.abspath(args.dataset_dir)
    train_images = os.path.join(dataset_dir, "images", "train")
    train_labels = os.path.join(dataset_dir, "labels", "train")
    val_images = os.path.join(dataset_dir, "images", "val")
    val_labels = os.path.join(dataset_dir, "labels", "val")

    dataset = DataFinalDetectionDataset(train_images, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if os.path.isdir(val_images) and os.path.isdir(val_labels):
        val_dataset = DataFinalDetectionDataset(val_images, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    print(f"Training images: {len(dataset)}")

    # 2 foreground classes + background.
    num_classes = 3
    model = get_model(num_classes=num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    run_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    losses = []
    best_loss = float("inf")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()
        losses.append(avg_loss)

        print(f"Epoch {epoch}/{args.epochs} complete | avg loss: {avg_loss:.4f}")

        latest_path = os.path.join(run_dir, "faster_rcnn_latest.pt")
        torch.save(model.state_dict(), latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(run_dir, "faster_rcnn_best.pt")
            torch.save(model.state_dict(), best_path)

    eval_metrics = {
        "mAP_50_95": None,
        "MAP_50": None,
        "MAP_75": None,
    }
    if val_loader is not None:
        print("Running validation metrics...")
        eval_metrics = evaluate_map(model, val_loader, device, num_classes=num_classes)
        print(
            "Validation metrics | "
            f"mAP_50_95: {eval_metrics['mAP_50_95']:.4f} | "
            f"MAP_50: {eval_metrics['MAP_50']:.4f} | "
            f"MAP_75: {eval_metrics['MAP_75']:.4f}"
        )

    summary = {
        "dataset_dir": dataset_dir,
        "run_name": args.run_name,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "best_loss": best_loss,
        "dataset_stats": {
            "total_images": dataset.stats.total_images,
            "missing_label_files": dataset.stats.missing_label_files,
            "dropped_bad_lines": dataset.stats.dropped_bad_lines,
            "dropped_bad_boxes": dataset.stats.dropped_bad_boxes,
        },
        "mAP_50_95": eval_metrics["mAP_50_95"],
        "MAP_50": eval_metrics["MAP_50"],
        "MAP_75": eval_metrics["MAP_75"],
    }

    summary_path = os.path.join(run_dir, "training_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Copy best model to models/rcnn/ and rename
    rcnn_dir = os.path.dirname(__file__)
    best_model_src = os.path.join(run_dir, "faster_rcnn_best.pt")
    best_model_dst = os.path.join(rcnn_dir, "rcnn_untuned.pt")
    shutil.copy(best_model_src, best_model_dst)
    
    print("Training complete.")
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()