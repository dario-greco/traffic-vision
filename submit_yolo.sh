#!/bin/bash
#SBATCH --job-name=yolo_a100
#SBATCH --partition=gpu_a100      # Targeting the A100 nodes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # 4 CPUs for data loading
#SBATCH --mem=32G                 # 32GB RAM
#SBATCH --time=05:00:00           # 5 hours is plenty of time
#SBATCH --output=logs/yolo_final_%j.log

echo "Job started on $(hostname)"
export PYTHONPATH=$PYTHONPATH:.

# Run the script using uv (unbuffered output so you can watch the logs live)
uv run python -u yolo/train_yolo_model.py

echo "Job finished"