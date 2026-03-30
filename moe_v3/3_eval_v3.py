#!/usr/bin/env python3
import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
TEST_SRC_IMG = BASE_DIR / "yolo_dataset" / "images" / "test"
TEST_SRC_LBL = BASE_DIR / "yolo_dataset" / "labels" / "test"

ROUTED_TEST_DIR = BASE_DIR / "moe_v3" / "test_routed"
EXPERT_DAY_DIR = ROUTED_TEST_DIR / "expert_day"
EXPERT_NIGHT_DIR = ROUTED_TEST_DIR / "expert_night"

# The empirical threshold we just calculated
BRIGHTNESS_THRESHOLD = 80.50

def calculate_brightness(image_path: Path) -> float:
    img = cv2.imread(str(image_path))
    if img is None: return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,2].mean()

def setup_test_directories():
    if ROUTED_TEST_DIR.exists():
        shutil.rmtree(ROUTED_TEST_DIR)
    
    for exp_dir in [EXPERT_DAY_DIR, EXPERT_NIGHT_DIR]:
        (exp_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (exp_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

def create_test_yaml(expert_dir: Path, name: str):
    yaml_content = f"""path: {expert_dir.absolute()}
train: images/test  # Dummy to satisfy YOLO
val: images/test    # Dummy to satisfy YOLO
test: images/test

names:
  0: stop_sign
  1: traffic_light
"""
    yaml_path = expert_dir / f"{name}_test.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path

def main():
    print("="*50)
    print("🧠 Initiating MoE V3 (Illumination) Evaluation")
    print("="*50)
    
    setup_test_directories()
    
    test_images = list(TEST_SRC_IMG.glob("*.jpg"))
    if not test_images:
        print("❌ Error: No test images found!")
        return
        
    print(f"Routing {len(test_images)} natural test images based on T={BRIGHTNESS_THRESHOLD}...")
    
    count_day = 0
    count_night = 0
    
    # 1. Route the Data
    for img_path in test_images:
        val = calculate_brightness(img_path)
        
        if val >= BRIGHTNESS_THRESHOLD:
            dest = EXPERT_DAY_DIR
            count_day += 1
        else:
            dest = EXPERT_NIGHT_DIR
            count_night += 1
            
        shutil.copy(img_path, dest / "images" / "test" / img_path.name)
        lbl_path = TEST_SRC_LBL / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, dest / "labels" / "test" / lbl_path.name)
            
    print(f"✅ Routing Complete: {count_day} to Day Expert, {count_night} to Night Expert.\n")
    
    yaml_day = create_test_yaml(EXPERT_DAY_DIR, "expert_day")
    yaml_night = create_test_yaml(EXPERT_NIGHT_DIR, "expert_night")
    
    # 2. Evaluate Day Expert
    map50_day = 0.0
    if count_day > 0:
        print("☀️ Evaluating Day Expert...")
        model_day = YOLO(BASE_DIR / "models" / "experts" / "expert_day_v3.pt")
        metrics_day = model_day.val(data=str(yaml_day), split='test', project="runs/moe_v3", name="eval_expert_day")
        map50_day = metrics_day.box.map50
    
    # 3. Evaluate Night Expert
    map50_night = 0.0
    if count_night > 0:
        print("\n🌙 Evaluating Night Expert...")
        model_night = YOLO(BASE_DIR / "models" / "experts" / "expert_night_v3.pt")
        metrics_night = model_night.val(data=str(yaml_night), split='test', project="runs/moe_v3", name="eval_expert_night")
        map50_night = metrics_night.box.map50
    
    # 4. Calculate the Global MoE mAP (Weighted Average)
    total_images = count_day + count_night
    weight_day = count_day / total_images
    weight_night = count_night / total_images
    
    global_map50 = (map50_day * weight_day) + (map50_night * weight_night)
    
    print("\n" + "="*50)
    print("🏆 FINAL MoE V3 PERFORMANCE REPORT")
    print("="*50)
    if count_day > 0: print(f"☀️ Day Expert mAP@50:   {map50_day:.4f}  (on {count_day} images)")
    if count_night > 0: print(f"🌙 Night Expert mAP@50: {map50_night:.4f}  (on {count_night} images)")
    print("-" * 50)
    print(f"🌟 Global MoE mAP@50:     {global_map50:.4f}")
    print("="*50)
    print("Compare this against your Baseline Generalist score of 0.5187!")

if __name__ == "__main__":
    main()