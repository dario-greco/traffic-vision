#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASELINE_YAML = BASE_DIR / "yolo_dataset" / "dataset.yaml"  # Fixed typo

def main():
    print("="*50)
    print("🔧 Training BASELINE (Generalist Model)")
    print("="*50)
    
    if not BASELINE_YAML.exists():
        print(f"❌ Error: Cannot find dataset config at {BASELINE_YAML}")
        return

    model = YOLO('yolov8n.pt')
    
    # Absolute path to prevent global settings hijack
    project_dir = BASE_DIR / "runs" / "baseline"
    
    results = model.train(
        data=str(BASELINE_YAML),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        project=str(project_dir),
        name="yolov8n_baseline"
    )
    
    # Harvest the best weights to your stable models folder
    best_weights = project_dir / "yolov8n_baseline" / "weights" / "best.pt"
    save_dir = BASE_DIR / "models" / "baselines"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = save_dir / "generalist_baseline.pt"
    if best_weights.exists():
        shutil.copy(best_weights, final_path)
        print(f"✅ Saved baseline weights to {final_path}")
    else:
        print("⚠️ Warning: best.pt not found. Training might have failed.")

    # Evaluate on the hold-out test set
    print("\n📊 Evaluating Baseline on Full Test Set...")
    metrics = model.val(
        data=str(BASELINE_YAML), 
        split='test',
        project=str(project_dir),
        name="yolov8n_test_eval"
    )
    
    baseline_map50 = metrics.box.map50
    
    print("\n" + "="*50)
    print("🏆 BASELINE PERFORMANCE REPORT")
    print("="*50)
    print(f"🌟 Global Baseline mAP@50: {baseline_map50:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()