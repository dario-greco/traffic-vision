import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from scripts.rcnn_train import DataFinalDetectionDataset, get_model, collate_fn, train_one_epoch


# We want to re-train the model on the full train + val set before the final test set eval. 
def train_production_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "runs/detect/rcnn_runs/production"
    os.makedirs(output_dir, exist_ok=True)

    # Hardcoded Champion Hyperparameters (From Trial 4)
    LR = 0.01
    STEP_SIZE = 3
    WEIGHT_DECAY = 0.0001
    GAMMA = 0.1
    EPOCHS = 7
    BATCH_SIZE = 16
    NUM_WORKERS = 8

    # Paths
    base_dir = "data_final"
    train_img = os.path.join(base_dir, "images", "train")
    train_lab = os.path.join(base_dir, "labels", "train")
    val_img = os.path.join(base_dir, "images", "val")
    val_lab = os.path.join(base_dir, "labels", "val")

    # Create individual datasets and merge them virtually
    train_ds = DataFinalDetectionDataset(train_img, train_lab)
    val_ds = DataFinalDetectionDataset(val_img, val_lab)
    combined_ds = ConcatDataset([train_ds, val_ds])

    print(f"Training on combined Train + Val sets: {len(combined_ds)} total images.")

    loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Model & Optimizer
    model = get_model(num_classes=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Training Loop (Fixed at 7 Epochs). 
    for epoch in range(1, EPOCHS + 1):
        print(f"Starting Epoch {epoch}/{EPOCHS}...")
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()
        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

    # Save Final Production Model
    final_model_path = os.path.join(output_dir, "FINAL_PRODUCTION_MODEL.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining Complete! Production model saved to: {final_model_path}")

if __name__ == "__main__":
    train_production_model()