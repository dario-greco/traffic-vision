from ultralytics import YOLO
import os
import shutil

# base directory = project root (parent of this scripts folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Load a pre-trained YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# 2. Get the absolute path to the dataset.yaml
yaml_path = os.path.join(BASE_DIR, "configs", "bdd10k_augment.yaml")

# 3. Get the yolo directory path for output
yolo_dir = os.path.dirname(__file__)

# 4. Start training
print("Starting YOLO training...")
results = model.train(
    data=yaml_path,
    epochs=20,             
    imgsz=640,             
    batch=16,              
    project=yolo_dir,
    name="traffic_model"   
)

# 5. Copy best model to yolo/ folder and rename
best_model_src = os.path.join(yolo_dir, "traffic_model", "weights", "best.pt")
best_model_dst = os.path.join(yolo_dir, "yolo_untrained.pt")
shutil.copy(best_model_src, best_model_dst)

print("Training complete! Model saved to yolo/yolo_untrained.pt")