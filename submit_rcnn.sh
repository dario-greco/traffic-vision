#!/bin/bash
#SBATCH --job-name=rcnn_hpo_a100
#SBATCH --partition=gpu_a100      
#SBATCH --gres=gpu:1              # Slurm will pick one of the 4 free cards
#SBATCH --cpus-per-task=8         
#SBATCH --mem=64G                 
#SBATCH --time=16:00:00           
#SBATCH --output=logs/rcnn_hpo_%j.log

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:.

# Launch the HPO coordinator
time uv run python -u scripts/rcnn_hpo.py

echo "Job finished at $(date)"