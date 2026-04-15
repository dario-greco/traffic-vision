from ultralytics import YOLO
import os

# 1. Using the correct path you found
model_path = "runs/detect/baseline/traffic_model/weights/best.pt" 
    
# 2. Get the absolute path to your dataset config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
yaml_path = os.path.join(BASE_DIR, "configs", "bdd10k_augment.yaml")

# 3. Load the model and evaluate
print(f"Loading weights from: {model_path}")
best_model = YOLO(model_path)

print("\n--- YOLOv8n FINAL TEST METRICS ---")
# Run validation on the 'test' split
test_metrics = best_model.val(data=yaml_path, split="test")

print("\n--- SUMMARY ---")
print(f"mAP_50_95 : {test_metrics.box.map:.4f}")
print(f"mAP_50    : {test_metrics.box.map50:.4f}")
print(f"mAP_75    : {test_metrics.box.map75:.4f}")