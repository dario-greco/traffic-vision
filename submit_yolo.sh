#!/bin/bash
#SBATCH --job-name=yolo_l40
#SBATCH --partition=gpu_l40s
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # 4 CPUs for data loading
#SBATCH --mem=32G                 # 32GB RAM
#SBATCH --time=05:00:00           # 5 hours is plenty of time
#SBATCH --output=logs/yolo_train_a100_%j.log

echo "Job started on $(hostname)"

# 1. Train the Clear Expert (Baseline)
uv run python -c "from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
model.train(data='yolo_dataset/dataset.yaml', epochs=10, imgsz=640, project='experts', name='clear_expert')"

# 2. Train the Blur Expert (Specialist)
uv run python -c "from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
model.train(data='yolo_dataset_blurred/dataset_blurred.yaml', epochs=10, imgsz=640, project='experts', name='blur_expert')"

echo "Job finished"

