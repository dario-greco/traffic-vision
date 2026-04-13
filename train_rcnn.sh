#!/bin/bash
#SBATCH --job-name=rcnn_a100
#SBATCH --partition=gpu_a100      # Targeting the A100 nodes
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # 8 CPUs for data loading
#SBATCH --mem=32G                 # 32GB RAM
#SBATCH --time=12:00:00           # 12 hours (R-CNN takes longer than YOLO)
#SBATCH --output=logs/rcnn_train_a100_%j.log

echo "Job started on $(hostname)"

# Run the script using uv (unbuffered output so you can watch the logs live)
uv run python -u scripts/4_train_rcnn_model.py \
	--dataset-dir data_final \
	--run-name traffic_model_rcnn_data_final

echo "Job finished"
