from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# 1. Locate the weights from your Slurm run
# These should appear in your 'experts' folder
CLEAR_PATH = Path("experts/clear_expert/weights/best.pt")
BLUR_PATH = Path("experts/blur_expert/weights/best.pt")

if not (CLEAR_PATH.exists() and BLUR_PATH.exists()):
    print("❌ Weights not found. Wait for the Slurm job to finish!")
    exit()

# 2. Load the Champions
clear_model = YOLO(CLEAR_PATH)
blur_model = YOLO(BLUR_PATH)

# 3. Define the Test Environments
test_tasks = [
    {"name": "Clear Test Set", "data": "yolo_dataset/dataset.yaml"},
    {"name": "Blurred Test Set", "data": "yolo_dataset_blurred/dataset_blurred.yaml"}
]

results = []

for task in test_tasks:
    print(f"\nEvaluating on: {task['name']}")
    
    # Audit Clear Model
    res_c = clear_model.val(data=task['data'], split='test', plots=False)
    # Audit Blur Expert
    res_b = blur_model.val(data=task['data'], split='test', plots=False)
    
    results.append({
        "Scenario": task['name'],
        "Clear Model mAP50": res_c.results_dict['metrics/mAP50(B)'],
        "Blur Expert mAP50": res_b.results_dict['metrics/mAP50(B)']
    })

# 4. The Grand Reveal
df = pd.DataFrame(results)
print("\n" + "="*40)
print("EXPERIMENT RESULTS: CLEAR VS. BLUR")
print("="*40)
print(df.to_string(index=False))
df.to_csv("final_moe_results.csv", index=False)