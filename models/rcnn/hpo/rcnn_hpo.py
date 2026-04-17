import os
import torch
import json
import itertools
import pandas as pd
from datetime import datetime
# Assuming your previous training logic is in scripts/rcnn_train.py
from models.rcnn.rcnn_pipeline import train_model, get_args

def run_hpo():
    # 1. Define the Search Grid
    grid = {
        'lr': [0.005, 0.01],
        'step_size': [3, 5],
        'weight_decay': [0.0001, 0.0005],
        'gamma': [0.1] 
    }

    # Generate all combinations
    keys, values = zip(*grid.items())
    trials = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results_list = []
    best_map = -1.0
    best_trial_name = ""
    
    print(f"Starting HPO Grid Search: {len(trials)} total trials.")

    for i, params in enumerate(trials):
        trial_name = f"trial_{i}_lr{params['lr']}_step{params['step_size']}_wd{params['weight_decay']}"
        print(f"\n--- [Trial {i+1}/{len(trials)}]: {trial_name} ---")
        
        # Setup arguments 
        args = get_args()
        args.run_name = os.path.join("hpo_grid", trial_name)
        args.lr = params['lr']
        args.step_size = params['step_size']
        args.weight_decay = params['weight_decay']
        args.gamma = params['gamma']
        args.batch_size = 16 # Optimized batch size for L40S
        args.num_workers = 8
        args.epochs = 15     

        # Run the training
        metrics = train_model(args)
        
        # Track results
        params['mAP_50_95'] = metrics.get('mAP_50_95', 0)
        params['mAP_50'] = metrics.get('MAP_50', 0)
        params['final_loss'] = metrics.get('final_loss', 0)
        results_list.append(params)
        
        # Save the absolute "Champion" model
        if params['mAP_50_95'] > best_map:
            best_map = params['mAP_50_95']
            best_trial_name = trial_name
            # Copy the best model to a central location
            best_model_path = os.path.join("runs/detect/rcnn_runs/hpo_grid", trial_name, "faster_rcnn_best.pt")
            target_path = "runs/detect/rcnn_runs/hpo_grid/CHAMPION_model.pt"
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy(best_model_path, target_path)
                print(f"New Champion found! Saved to {target_path}")

    # 2. Save HPO Summary
    df = pd.DataFrame(results_list)
    df.to_csv("runs/detect/rcnn_runs/hpo_grid/hpo_summary.csv", index=False)
    print("\nGrid Search Complete. Results saved to hpo_summary.csv")

if __name__ == "__main__":
    run_hpo()