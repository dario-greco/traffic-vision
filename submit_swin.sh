#!/bin/bash -l
#SBATCH --job-name=swin_transformer
#SBATCH --gres=gpu:l40s:1         # Request 1 GPU
#SBATCH --cpus-per-task=8         # 8 CPUs for data loading
#SBATCH --mem=48G                 # 48GB RAM
#SBATCH --time=12:00:00           # 12 hours for 20 epochs
#SBATCH --output=logs/swin_transformer_train_%j.log

echo "Job started on $(hostname)"

# Run the script using uv (unbuffered output so you can watch the logs live)
uv run python -u scripts/2.1_swin_transformer_without_ft.py

echo "Job finished"
