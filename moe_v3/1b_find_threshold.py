#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DAY_DIR = BASE_DIR / "yolo_dataset" / "images" / "train"
NIGHT_DIR = BASE_DIR / "moe_v3" / "yolo_dataset_night" / "images" / "train"

def calculate_brightness(image_path: Path) -> float:
    """Calculates the average HSV Value (brightness) of an image."""
    img = cv2.imread(str(image_path))
    if img is None: return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,2].mean()

def main():
    print("="*50)
    print("🔍 Empirically Searching for Optimal Illumination Threshold")
    print("="*50)

    day_images = list(DAY_DIR.glob("*.jpg"))
    night_images = list(NIGHT_DIR.glob("*.jpg"))

    if not day_images or not night_images:
        print("❌ Error: Could not find images in both Day and Night directories.")
        print("Did you run 1_generate_night_data.py?")
        return

    print(f"Extracting brightness values from {len(day_images)} Day images...")
    day_vals = np.array([calculate_brightness(p) for p in day_images])
    
    print(f"Extracting brightness values from {len(night_images)} Night images...")
    night_vals = np.array([calculate_brightness(p) for p in night_images])

    all_vals = np.concatenate([day_vals, night_vals])
    global_median = np.median(all_vals)

    print("\nRunning linear search for maximum accuracy...")
    best_t = 0
    best_acc = 0.0

    # Search every possible brightness value from 0 to 255
    for t in np.arange(0, 255, 0.5):
        # Accuracy = (True Positives + True Negatives) / Total Population
        # Day should be >= T, Night should be < T
        correct_day = np.sum(day_vals >= t)
        correct_night = np.sum(night_vals < t)
        
        acc = (correct_day + correct_night) / len(all_vals)

        if acc > best_acc:
            best_acc = acc
            best_t = t

    print("\n" + "="*50)
    print("📊 EMPIRICAL THRESHOLD RESULTS")
    print("="*50)
    print(f"Global Median (Baseline Split):  {global_median:.2f}")
    print(f"Optimal Separating Threshold:    {best_t:.2f}")
    print(f"Router Accuracy at Optimal T:    {best_acc * 100:.2f}%")
    print("="*50)
    
    if best_acc < 0.90:
        print("\n⚠️ Note: The distributions overlap significantly. The router will make mistakes.")
    else:
        print("\n✅ The distributions are cleanly separated! Use the Optimal T in your inference script.")

if __name__ == "__main__":
    main()