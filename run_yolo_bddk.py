from ultralytics import YOLO
import os

def main():
    # 1. Load the model (YOLOv8 Nano is best for the smoke test)
    # It will automatically download 'yolov8n.pt' if not found in the directory
    model = YOLO("yolov8n.pt")

    # 2. Define the path to your YAML file
    # Using the absolute path we verified earlier
    yaml_path = "/work/d2greco/traffic-vision/bdd10k.yaml"

    print(f"🚀 Starting training with data from: {yaml_path}")

    # 3. Train the model
    # We're using 30 epochs and a batch size of 32 for the L40S GPU
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=32,
        project="bdd_research",
        name="initial_10k_subset",
        device=0,         # Use the first available GPU
        workers=8,        # Match this to your Slurm cpus-per-task
        exist_ok=True     # Overwrite if the experiment name already exists
    )

    print("✅ Training complete. Results saved to 'bdd_research/initial_10k_subset'")

if __name__ == "__main__":
    main()