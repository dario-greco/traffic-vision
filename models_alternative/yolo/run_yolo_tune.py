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

    print(f"Starting model TUNING with clean data from: {yaml_path}")

    # 3. Train the model
    # A batch size of 32 works for the L40S GPU
    model.tune(
        data=yaml_path,
        epochs=30,
        iterations=20,
        optimizer="AdamW",
        project="bdd_research",
        name="hpo_dropout_20iter",
        batch=32,
        imgsz=640,
        device=0,
        workers=8,
        exist_ok=True,
        # Starting search values
        dropout=0.1,      
        weight_decay=0.0005,
        val=True
    )

    # Check yolo global setting to see teh exact folder path
    print("Training complete. Results saved to 'bdd_research/hpo_dropout_20iter'")

if __name__ == "__main__":
    main()