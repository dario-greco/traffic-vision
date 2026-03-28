#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# submit_dfine.sh  —  Slurm job script for D-FINE-M training.
#
# Mirrors the style of submit_yolo.sh at the repo root.
#
# Submit with:
#   sbatch D-FINE/submit_dfine.sh
# or from inside D-FINE/:
#   sbatch submit_dfine.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=dfine_m_traffic
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dfine_train_%j.log

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID : $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo ""

# Create logs directory if needed
mkdir -p logs

# Navigate to the D-FINE folder (works whether submitted from repo root or D-FINE/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Run the full setup + training pipeline
bash setup_and_train.sh

echo ""
echo "Job finished at $(date)"
