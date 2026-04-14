import os
import json
import torch
from torch.utils.data import DataLoader
from scripts.rcnn_train import DataFinalDetectionDataset, get_model, collate_fn, evaluate_map

def evaluate_test_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "runs/detect/rcnn_runs/production/FINAL_PRODUCTION_MODEL.pt"
    output_dir = "runs/detect/rcnn_runs/production"
    
    # Paths
    test_img = "data_final/images/test"
    test_lab = "data_final/labels/test"

    print("Loading Test Dataset...")
    test_ds = DataFinalDetectionDataset(test_img, test_lab)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, 
                             num_workers=8, collate_fn=collate_fn)

    print("Loading Production Model...")
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    print("Starting Test Set Evaluation ...")
    metrics = evaluate_map(model, test_loader, device, num_classes=3)
    
    print("\n--- FINAL TEST SET RESULTS ---")
    print(f"mAP_50_95 : {metrics.get('mAP_50_95', 0):.4f}")
    print(f"mAP_50    : {metrics.get('MAP_50', 0):.4f}") 
    print(f"mAP_75    : {metrics.get('MAP_75', 0):.4f}")

    # Save results
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    evaluate_test_set()