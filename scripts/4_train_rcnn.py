"""
Faster R-CNN Training Script for Traffic Sign and Light Detection
Includes:
- YOLO → Faster R-CNN annotation conversion
- Training loop
- COCO-style mAP evaluation
- Precision–Recall curve
- Training loss plot
"""

import os
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_DIR = os.path.join(BASE_DIR, "yolo_dataset")
YOLO_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
YOLO_LABELS_DIR = os.path.join(DATASET_DIR, "labels")

OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "detect", "rcnn_runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# DATASET
# =========================================================

class COCODataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = torchvision.io.read_image(img_path).float() / 255.0

        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):

            with open(label_path) as f:
                lines = f.readlines()

            h, w = image.shape[1], image.shape[2]

            for line in lines:

                cls, xc, yc, bw, bh = map(float, line.split())

                xc *= w
                yc *= h
                bw *= w
                bh *= h

                x_min = xc - bw / 2
                y_min = yc - bh / 2
                x_max = xc + bw / 2
                y_max = yc + bh / 2

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls) + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================================================
# MODEL
# =========================================================

def get_rcnn_model(num_classes):

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model


# =========================================================
# TRAINING
# =========================================================

def train_one_epoch(model, optimizer, loader, device, epoch):

    model.train()

    total_loss = 0

    for i, (images, targets) in enumerate(loader):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % 10 == 0:
            print(
                f"Epoch {epoch} | Batch {i+1}/{len(loader)} | Loss {losses.item():.4f}"
            )

    return total_loss / len(loader)


# =========================================================
# EVALUATION
# =========================================================

def evaluate_map(model, loader, device):

    model.eval()

    metric = MeanAveragePrecision(
        iou_type="bbox",
        extended_summary=True   # IMPORTANT: enables precision tensor
    )

    with torch.no_grad():

        for images, targets in loader:

            images = [img.to(device) for img in images]

            outputs = model(images)

            preds = []
            gts = []

            for output, target in zip(outputs, targets):

                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu(),
                })

                gts.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })

            metric.update(preds, gts)

    return metric.compute()


# =========================================================
# PRECISION–RECALL PLOT
# =========================================================

def plot_precision_recall(results, save_path):

    precision = results["precision"]

    # dims: [IoU, recall, class, area, maxDet]
    precision_curve = precision[0, :, 0, 0, 2].cpu().numpy()

    recall = np.linspace(0, 1, len(precision_curve))

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision_curve)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)

    plt.savefig(save_path)


# =========================================================
# MAIN
# =========================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = 3
    batch_size = 4
    num_epochs = 10

    lr = 0.005
    weight_decay = 0.0005

    print("Loading datasets...")

    train_dataset = COCODataset(
        os.path.join(YOLO_IMAGES_DIR, "train"),
        os.path.join(YOLO_LABELS_DIR, "train"),
    )

    val_dataset = COCODataset(
        os.path.join(YOLO_IMAGES_DIR, "val"),
        os.path.join(YOLO_LABELS_DIR, "val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    model = get_rcnn_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    losses = []

    print("\nStarting training...\n")

    for epoch in range(num_epochs):

        loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch + 1,
        )

        losses.append(loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} complete | avg loss {loss:.4f}\n")

    print("Evaluating model...")

    map_results = evaluate_map(model, val_loader, device)

    print("\n===== Detection Metrics =====")
    print(f"mAP (AP@0.50:0.95): {map_results['map']:.4f}")
    print(f"AP@0.50: {map_results['map_50']:.4f}")
    print(f"AP@0.75: {map_results['map_75']:.4f}")
    print("=============================\n")

    save_dir = os.path.join(OUTPUT_DIR, "traffic_model_rcnn")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "faster_rcnn_best.pt"),
    )

    results_dict = {
        "mAP_50_95": float(map_results["map"]),
        "AP_50": float(map_results["map_50"]),
        "AP_75": float(map_results["map_75"]),
        "losses": losses,
    }

    with open(os.path.join(save_dir, "training_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    # training loss plot
    plt.figure(figsize=(10,6))
    plt.plot(range(1, num_epochs+1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_loss.png"))

    # precision–recall curve
    plot_precision_recall(
        map_results,
        os.path.join(save_dir, "precision_recall_curve.png"),
    )

    print("Results saved to:", save_dir)


if __name__ == "__main__":
    main()