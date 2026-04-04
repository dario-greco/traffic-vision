from ultralytics import YOLO
import os

def main():
    # 1. Load the model 
    # It will automatically download 'yolov8n.pt' if not found in the directory
    model = YOLO("yolov8n.pt")

    # 2. Define the path to your YAML file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    yaml_path = os.path.join(root_dir, "configs", "bdd10k.yaml")

    print(f"Starting training with data from: {yaml_path}")

    # 3. Train the model
    # A batch size of 32 works for the L40S GPU
    results = model.train(
        data=yaml_path,
        epochs=5,
        imgsz=640,
        batch=32,
        project="bdd_research",
        name="initial_10k_subset_test",
        device=0,         # Use the first available GPU
        workers=8,        # Match this to your Slurm cpus-per-task
        exist_ok=True     # Overwrite 
    )

    # Check yolo global setting to see teh exact folder path
    print("Training complete. Results saved to 'bdd_research/initial_10k_subset'")

if __name__ == "__main__":
    main()