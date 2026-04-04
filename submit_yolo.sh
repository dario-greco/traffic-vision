#!/bin/bash
#SBATCH --job-name=bdd_yolo_train
#SBATCH --partition=gpu_l40s
#SBATCH --gres=gpu:1              
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32G                 
#SBATCH --time=02:00:00           
#SBATCH --output=logs/bdd_train_%j.log

echo "========================================"
echo "Job: BDD 10k Test"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# Execute the training script via uv
uv run python run_yolo_bddk.py

echo "========================================"
echo "Job finished at $(date)"