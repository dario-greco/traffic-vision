#!/bin/bash
#SBATCH --job-name=rcnn_train
#SBATCH --partition=gpu_l40s      
#SBATCH --gres=gpu:1              
#SBATCH --cpus-per-task=8         # Matches the 'num_workers' in 4_rcnn.py
#SBATCH --mem=32G                 
#SBATCH --time=08:00:00           # Faster R-CNN takes longer than YOLO; 8h is safer
#SBATCH --output=logs/rcnn_train_%j.log

echo "Job started on $(hostname)"

# Run the RCNN script
# -u ensures Python doesn't buffer logs, so you see progress immediately
uv run python -u scripts/4_rcnn.py --batch-size 4 --epochs 20

echo "Job finished"