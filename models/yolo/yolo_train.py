from ultralytics import YOLO
import os

# base directory = project root (parent of this scripts folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Load a pre-trained YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# 2. Get the absolute path to the dataset.yaml
yaml_path = os.path.join(BASE_DIR, "configs", "bdd10k_augment.yaml")

# 3. Start training
print("Starting YOLO training...")
results = model.train(
    data=yaml_path,
    epochs=20,             
    imgsz=640,             
    batch=16,              
    project="baseline",
    name="traffic_model"   
)
print("Training complete! Model saved in baseline/traffic_model/weights/best.pt")
