#!/bin/bash
#SBATCH --job-name=rcnn_bs16
#SBATCH --partition=gpu_l40s      
#SBATCH --gres=gpu:1              
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32G                 
#SBATCH --time=04:00:00           # Reduced time as this should be faster
#SBATCH --output=logs/rcnn_bs16_%j.log

echo "Job started on $(hostname) at $(date)"

# Use 'time' to measure the duration of the training process
# Added --num-workers 8, --lr 0.01, and a unique --run-name
time uv run python -u scripts/4_rcnn.py \
    --batch-size 16 \
    --num-workers 8 \
    --lr 0.01 \
    --run-name rcnn_batch16_v1

echo "Job finished at $(date)"