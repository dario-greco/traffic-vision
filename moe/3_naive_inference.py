#!/usr/bin/env python3
import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
TEST_SRC_IMG = BASE_DIR / "yolo_dataset" / "images" / "test"
TEST_SRC_LBL = BASE_DIR / "yolo_dataset" / "labels" / "test"

ROUTED_TEST_DIR = BASE_DIR / "moe" / "test_routed"
EXPERT_0_DIR = ROUTED_TEST_DIR / "expert_0_blur"
EXPERT_1_DIR = ROUTED_TEST_DIR / "expert_1_clear"

# The hardcoded threshold from your training distribution
LAPLACIAN_THRESHOLD = 3058.97

def calculate_laplacian_variance(image_path: Path) -> float:
    img = cv2.imread(str(image_path))
    if img is None: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def setup_test_directories():
    """Clears and prepares the temporary routing directories."""
    if ROUTED_TEST_DIR.exists():
        shutil.rmtree(ROUTED_TEST_DIR)
    
    for exp_dir in [EXPERT_0_DIR, EXPERT_1_DIR]:
        (exp_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (exp_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

def create_test_yaml(expert_dir: Path, name: str):
    yaml_content = f"""path: {expert_dir.absolute()}
train: images/test
val: images/test
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
    print("🧠 Initiating Naive MoE Evaluation")
    print("="*50)
    
    setup_test_directories()
    
    test_images = list(TEST_SRC_IMG.glob("*.jpg"))
    if not test_images:
        print("❌ Error: No test images found!")
        return
        
    print(f"Routing {len(test_images)} test images based on T={LAPLACIAN_THRESHOLD}...")
    
    count_0 = 0
    count_1 = 0
    
    # 1. Route the Data
    for img_path in test_images:
        var = calculate_laplacian_variance(img_path)
        
        if var < LAPLACIAN_THRESHOLD:
            dest = EXPERT_0_DIR
            count_0 += 1
        else:
            dest = EXPERT_1_DIR
            count_1 += 1
            
        shutil.copy(img_path, dest / "images" / "test" / img_path.name)
        lbl_path = TEST_SRC_LBL / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, dest / "labels" / "test" / lbl_path.name)
            
    print(f"✅ Routing Complete: {count_0} to Blur Expert, {count_1} to Clear Expert.\n")
    
    yaml_0 = create_test_yaml(EXPERT_0_DIR, "expert_0")
    yaml_1 = create_test_yaml(EXPERT_1_DIR, "expert_1")
    
    # 2. Evaluate Expert 0
    print("📊 Evaluating Expert 0 (Blur Domain)...")
    model_0 = YOLO(BASE_DIR / "models" / "experts" / "expert_0_blur_naive.pt")
    metrics_0 = model_0.val(data=str(yaml_0), split='test', project="runs/moe", name="eval_expert_0")
    map50_0 = metrics_0.box.map50
    
    # 3. Evaluate Expert 1
    print("\n📊 Evaluating Expert 1 (Clear Domain)...")
    model_1 = YOLO(BASE_DIR / "models" / "experts" / "expert_1_clear_naive.pt")
    metrics_1 = model_1.val(data=str(yaml_1), split='test', project="runs/moe", name="eval_expert_1")
    map50_1 = metrics_1.box.map50
    
    # 4. Calculate the Global MoE mAP (Weighted Average)
    total_images = count_0 + count_1
    weight_0 = count_0 / total_images
    weight_1 = count_1 / total_images
    
    global_map50 = (map50_0 * weight_0) + (map50_1 * weight_1)
    
    print("\n" + "="*50)
    print("🏆 FINAL MoE PERFORMANCE REPORT")
    print("="*50)
    print(f"Expert 0 (Blur)  mAP@50:  {map50_0:.4f}  (on {count_0} images)")
    print(f"Expert 1 (Clear) mAP@50:  {map50_1:.4f}  (on {count_1} images)")
    print("-" * 50)
    print(f"🌟 Global MoE mAP@50:     {global_map50:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()