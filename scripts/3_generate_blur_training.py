import os
import cv2
import shutil
from pathlib import Path

# Set your project root
BASE_DIR = Path.cwd()
SRC_DIR = BASE_DIR / "yolo_dataset"
DEST_DIR = BASE_DIR / "yolo_dataset_blurred"

def generate_blurred_twin(kernel_size=(15, 15)):
    if DEST_DIR.exists():
        print(f"Cleaning old blurred data at {DEST_DIR}...")
        shutil.rmtree(DEST_DIR)
    
    # Mirror the directory structure
    for split in ['train', 'val', 'test']:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        print(f"Processing {split} split...")
        img_path = SRC_DIR / "images" / split
        
        for img_file in img_path.glob("*.jpg"):
            # 1. Read and Blur Image
            img = cv2.imread(str(img_file))
            blurred = cv2.GaussianBlur(img, kernel_size, 0)
            
            # 2. Save Blurred Image
            cv2.imwrite(str(DEST_DIR / "images" / split / img_file.name), blurred)
            
            # 3. Copy Label (Coordinates remain identical!)
            label_file = SRC_DIR / "labels" / split / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, DEST_DIR / "labels" / split / label_file.name)

    # 4. Create the new dataset.yaml
    yaml_content = f"""
path: {DEST_DIR.absolute()}
train: images/train
val: images/val
test: images/test

names:
  0: stop sign
  1: traffic light
"""
    with open(DEST_DIR / "dataset_blurred.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"\n✅ Success! Blurred dataset and YAML created at: {DEST_DIR}")

if __name__ == "__main__":
    generate_blurred_twin()