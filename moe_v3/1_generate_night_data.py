#!/usr/bin/env python3
import os
import cv2
import shutil
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SRC_DIR = BASE_DIR / "yolo_dataset"
MOE_DIR = BASE_DIR / "moe_v3"
DEST_DIR = MOE_DIR / "yolo_dataset_night"

def simulate_night_vision(image_path: Path, darkness_factor=0.4):
    """Reduces the V channel in HSV space to simulate night/underexposed conditions."""
    img = cv2.imread(str(image_path))
    if img is None: return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Scale down the Value (brightness) channel
    hsv[:, :, 2] = hsv[:, :, 2] * darkness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def setup_directories():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    
    for split in ['train', 'val', 'test']:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def main():
    print("="*50)
    print("🌙 Generating Synthetic Night Domain (MoE V3)")
    print("="*50)
    
    MOE_DIR.mkdir(exist_ok=True)
    setup_directories()
    
    for split in ['train', 'val', 'test']:
        print(f"Processing '{split}' split...")
        src_img_dir = SRC_DIR / "images" / split
        src_lbl_dir = SRC_DIR / "labels" / split
        
        count = 0
        for img_file in src_img_dir.glob("*.jpg"):
            # 1. Synthesize Night Image
            night_img = simulate_night_vision(img_file)
            if night_img is not None:
                cv2.imwrite(str(DEST_DIR / "images" / split / img_file.name), night_img)
            
            # 2. Copy Exact Labels (Coordinates do not change at night)
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy(lbl_file, DEST_DIR / "labels" / split / lbl_file.name)
            count += 1
            
        print(f"  -> Generated {count} night images.")

    # 3. Create the YAML config for the Night Expert
    yaml_content = f"""path: {DEST_DIR.absolute()}
train: images/train
val: images/val
test: images/test

names:
  0: stop_sign
  1: traffic_light
"""
    yaml_path = DEST_DIR / "night_expert.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"\n✅ Success! Night dataset created at: {DEST_DIR}")

if __name__ == "__main__":
    main()