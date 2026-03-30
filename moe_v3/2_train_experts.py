#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Domain YAMLs
DAY_YAML = BASE_DIR / "yolo_dataset" / "dataset.yaml"
NIGHT_YAML = BASE_DIR / "moe_v3" / "yolo_dataset_night" / "night_expert.yaml"

def train_domain_expert(expert_yaml: Path, expert_name: str, save_name: str):
    print(f"\n{'='*50}")
    print(f"🚀 Training Domain Specialist: {expert_name.upper()}")
    print(f"Dataset: {expert_yaml}")
    print(f"{'='*50}\n")
    
    if not expert_yaml.exists():
        print(f"❌ Error: Cannot find dataset config at {expert_yaml}")
        return
        
    model = YOLO('yolov8n.pt')
    
    # Absolute path to prevent global settings hijack
    project_dir = BASE_DIR / "runs" / "moe_v3"
    
    results = model.train(
        data=str(expert_yaml),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        project=str(project_dir),
        name=expert_name
    )
    
    # Harvest the best weights to your stable models folder
    best_weights = project_dir / expert_name / "weights" / "best.pt"
    save_dir = BASE_DIR / "models" / "experts"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = save_dir / f"{save_name}.pt"
    if best_weights.exists():
        shutil.copy(best_weights, final_path)
        print(f"✅ Saved {save_name} weights to {final_path}")
    else:
        print(f"⚠️ Warning: best.pt not found for {expert_name}. Training might have failed.")

def main():
    print("Initializing MoE V3 (Illumination) Training Pipeline...")
    
    # 1. Train the Day Expert (Original Data)
    train_domain_expert(
        expert_yaml=DAY_YAML, 
        expert_name="expert_0_day", 
        save_name="expert_day_v3"
    )
    
    # 2. Train the Night Expert (Synthetic Dark Data)
    train_domain_expert(
        expert_yaml=NIGHT_YAML, 
        expert_name="expert_1_night", 
        save_name="expert_night_v3"
    )
    
    print("\n🎉 Both Illumination Experts have completed training!")
    print("Weights are secured in the models/experts/ directory.")

if __name__ == "__main__":
    main()