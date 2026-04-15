#!/bin/bash
#SBATCH --job-name=rcnn_train
#SBATCH --partition=gpu_a100      
#SBATCH --gres=gpu:1              # Slurm will pick one of the 4 free cards
#SBATCH --cpus-per-task=8         
#SBATCH --mem=64G                 
#SBATCH --time=16:00:00           
#SBATCH --output=logs/rcnn_full_eval%j.log

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:.

# Launch the HPO coordinator
# time uv run python -u scripts/rcnn_hpo.py

# Train the final model on Train + Val
uv run python -u scripts/final_train.py

# Immediately evaluate the newly saved model on the Test set
uv run python -u scripts/eval_test.py

echo "Job finished at $(date)"