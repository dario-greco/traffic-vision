import fiftyone as fo
import fiftyone.zoo as foz
import os
import sys

classes = ["stop sign", "traffic light"]
# Get the parent directory of the current script and then navigate to the "yolo_dataset" folder
export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolo_dataset"))

# 1. Safe-Guard: Check if the dataset already exists before doing anything
if os.path.exists(export_dir) and len(os.listdir(export_dir)) > 0:
    print(f"✅ Dataset already exists at: {export_dir}")
    print("Skipping download. (If you need fresh data, delete the 'yolo_dataset' folder and run this script again).")
    sys.exit() # Stops the script here so it doesn't run the download code below

# 2. Clean up any old datasets in FiftyOne to avoid conflicts
for name in ["coco-traffic-train", "coco-traffic-val"]:
    if name in fo.list_datasets():
        fo.delete_dataset(name)

print(f"No dataset found. Downloading and exporting to {export_dir}...")

# 3. Download and Export
print("\nProcessing Training split...")
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    classes=classes,
    max_samples=500,
    shuffle=True,    
    seed=42,
    dataset_name="coco-traffic-train", 
)

train_dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
    classes=classes,
)

print("\nProcessing Validation split...")
val_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    classes=classes,
    max_samples=100,
    shuffle=True,
    seed=42,
    dataset_name="coco-traffic-val", 
)

val_dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
    classes=classes,
)

print(f"\n Success! Data exported to {export_dir}")