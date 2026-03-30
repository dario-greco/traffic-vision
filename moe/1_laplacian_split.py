#!/usr/bin/env python3
import os
import cv2
import shutil
import numpy as np
from pathlib import Path

# --- Configuration ---
# Assuming you run this from inside the moe/ folder
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SRC_DATA = BASE_DIR / "yolo_dataset"
MOE_DIR = BASE_DIR / "moe"

EXPERT_0_DIR = MOE_DIR / "data_expert_0_blur"
EXPERT_1_DIR = MOE_DIR / "data_expert_1_clear"

def calculate_laplacian_variance(image_path: Path) -> float:
    """Calculates the variance of the Laplacian (standard focus/blur metric)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def setup_directories():
    """Wipe old MoE data and create fresh directory structures."""
    for expert_dir in [EXPERT_0_DIR, EXPERT_1_DIR]:
        if expert_dir.exists():
            shutil.rmtree(expert_dir)
        for split in ['train', 'val']:
            (expert_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (expert_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def create_yaml(expert_dir: Path, expert_name: str):
    """Generates the YOLO dataset.yaml for the specific expert."""
    yaml_content = f"""path: {expert_dir.absolute()}
train: images/train
val: images/val

names:
  0: stop_sign
  1: traffic_light
"""
    yaml_path = expert_dir / f"{expert_name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path

def main():
    print("Initializing Naive MoE Data Splitter...")
    setup_directories()

    train_images_dir = SRC_DATA / "images" / "train"
    train_images = list(train_images_dir.glob("*.jpg"))
    
    if not train_images:
        print(f"❌ No training images found in {train_images_dir}")
        return

    # 1. Calculate the Threshold (The "Naive Router") based ONLY on Train
    print(f"\nAnalyzing {len(train_images)} training images to find routing threshold...")
    variances = []
    for img_path in train_images:
        var = calculate_laplacian_variance(img_path)
        variances.append(var)

    # The Median is our threshold T
    threshold = np.median(variances)
    print(f"✅ Median Laplacian Variance (Threshold T) = {threshold:.2f}")

    # 2. Distribute Train and Val sets
    # Note: We do NOT process 'test' here. The test set stays untouched for evaluation.
    for split in ['train', 'val']:
        print(f"\nRouting '{split}' data...")
        src_img_dir = SRC_DATA / "images" / split
        src_lbl_dir = SRC_DATA / "labels" / split
        
        count_0 = 0
        count_1 = 0

        for img_path in src_img_dir.glob("*.jpg"):
            # Route based on variance
            var = calculate_laplacian_variance(img_path)
            if var < threshold:
                dest_root = EXPERT_0_DIR
                count_0 += 1
            else:
                dest_root = EXPERT_1_DIR
                count_1 += 1

            # Copy Image
            shutil.copy(img_path, dest_root / "images" / split / img_path.name)
            
            # Copy Label (if it exists)
            lbl_name = img_path.stem + ".txt"
            lbl_path = src_lbl_dir / lbl_name
            if lbl_path.exists():
                shutil.copy(lbl_path, dest_root / "labels" / split / lbl_name)

        print(f"  -> Expert 0 (Blur/Low-Var) received: {count_0} images")
        print(f"  -> Expert 1 (Clear/High-Var) received: {count_1} images")

    # 3. Create the YAML config files
    create_yaml(EXPERT_0_DIR, "expert_0")
    create_yaml(EXPERT_1_DIR, "expert_1")
    
    print("\n🎉 MoE Data Split Complete!")
    print(f"Threshold T={threshold:.2f} should be hardcoded into your future Router script.")

if __name__ == "__main__":
    main()