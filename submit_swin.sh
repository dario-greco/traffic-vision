#!/bin/bash -l
#SBATCH --job-name=swin_transformer
#SBATCH --gres=gpu:l40s:1         # Request 1 GPU
#SBATCH --cpus-per-task=8         # 4 CPUs for data loading
#SBATCH --mem=32G                 # 32GB RAM
#SBATCH --time=05:00:00           # 5 hours is plenty of time
#SBATCH --output=logs/swin_transformer_train_%j.log

echo "Job started on $(hostname)"

# Run the script using uv (unbuffered output so you can watch the logs live)
uv run python -u scripts/2.1_swin_transformer.py

echo "Job finished"
