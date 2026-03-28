import cv2
import os
import random
from glob import glob

# Pathing setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(BASE_DIR, "yolo_dataset", "images", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "blur_samples")

# 1. Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

def main():
    # 2. Find all training images
    images = glob(os.path.join(SOURCE_DIR, "*.jpg"))
    if not images:
        print("No images found! Make sure you ran the prepare_data script first.")
        return

    # 3. Select 5 random samples to test
    samples = random.sample(images, 5)
    print(f"Blurring {len(samples)} sample images...")

    for img_path in samples:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Apply a heavy Gaussian Blur (31x31 kernel for high visibility)
            blurred = cv2.GaussianBlur(img, (11, 11), 0)
            
            # Save to the NEW folder
            save_path = os.path.join(OUTPUT_DIR, f"blurred_{filename}")
            cv2.imwrite(save_path, blurred)
            print(f"Saved: {save_path}")

    print("\n✅ Done! Check the 'blur_samples' folder to see the results.")

if __name__ == "__main__":
    main()