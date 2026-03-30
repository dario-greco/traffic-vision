#!/usr/bin/env python3
import os
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MOE_DIR = BASE_DIR / "moe"

def train_expert(expert_yaml: Path, expert_name: str):
    print(f"\n{'='*50}")
    print(f"🚀 Training Specialist: {expert_name.upper()}")
    print(f"{'='*50}\n")
    
    # Always initialize a fresh YOLOv8 Nano model for each expert
    model = YOLO('yolov8n.pt')
    
    # Get folder paths 
    project_dir = BASE_DIR / "runs" / "moe"
    
    # Train the model
    # Note: We use patience=20 to stop early if it converges fast on the smaller dataset
    results = model.train(
        data=str(expert_yaml),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        project=str(project_dir), # <--- Force absolute path here
        name=expert_name
    )
    
    # Move the best weights to a stable location
    best_weights = project_dir / expert_name / "weights" / "best.pt"
    save_dir = BASE_DIR / "models" / "experts"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = save_dir / f"{expert_name}_naive.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, final_path)
        print(f"✅ Saved best weights to {final_path}")
    else:
        print("⚠️ Warning: best.pt not found. Training might have failed.")

def main():
    expert_0_yaml = MOE_DIR / "data_expert_0_blur" / "expert_0.yaml"
    expert_1_yaml = MOE_DIR / "data_expert_1_clear" / "expert_1.yaml"
    
    if not expert_0_yaml.exists() or not expert_1_yaml.exists():
        print("❌ Error: YAML files not found. Did you run 1_laplacian_split.py?")
        return

    # Train Expert 0 (Blur / Low-Variance)
    train_expert(expert_0_yaml, "expert_0_blur")
    
    # Train Expert 1 (Clear / High-Variance)
    train_expert(expert_1_yaml, "expert_1_clear")
    
    print("\n🎉 Both Naive Experts have completed training!")

if __name__ == "__main__":
    main()