from ultralytics import YOLO
import os

# base directory = project root (parent of this scripts folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Load a pre-trained YOLOv8 Nano model (fastest model, great for testing)
model = YOLO('yolov8n.pt')

# 2. Get the absolute path to the dataset.yaml file FiftyOne just created
# YOLO requires the absolute path to avoid directory confusion
yaml_path = os.path.join(BASE_DIR, "configs", "bdd10k_augment.yaml")

# 3. Start training
print("Starting YOLO training...")
results = model.train(
    data=yaml_path,
    epochs=10,             # Number of times it loops through the data 
    imgsz=640,             # Standard image size for YOLO
    batch=16,              # Number of images processed at once
    project="traffic_v8n_runs",# Folder where results will be saved
    name="traffic_model"   # Subfolder for this specific training run
)

print("Training complete! Model saved in traffic_v8n_runs/traffic_model/weights/best.pt")