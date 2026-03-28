#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_and_train.sh  —  One-shot D-FINE-M setup + fine-tuning.
#
# Run from the D-FINE/ directory:
#   bash setup_and_train.sh
#
# Or submit to Slurm (see submit_dfine.sh).
#
# What this script does:
#   1. Installs missing D-FINE Python dependencies into the project venv.
#   2. Prepares the COCO-format dataset from yolo_dataset/ (idempotent).
#   3. Downloads the D-FINE-M obj2coco pre-trained weights (skips if present).
#   4. Fine-tunes D-FINE-M on the traffic detection dataset.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="custom/configs/dfine_m_traffic.yml"
WEIGHTS="custom/weights/dfine_m_obj2coco.pth"
NPROC=1       # number of GPUs; increase on multi-GPU nodes
PORT=7777
SEED=0

# ── Resolve script directory so the script works when called from anywhere ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " D-FINE-M  ·  Traffic Detection  ·  Full Setup"
echo "=============================================="
echo "Working dir : $(pwd)"
echo "Config      : $CONFIG"
echo "Weights     : $WEIGHTS"
echo "GPUs        : $NPROC"
echo "Date        : $(date)"
echo ""

# ── Check D-FINE repo is cloned ───────────────────────────────────────────────
if [[ ! -f "train.py" || ! -d "src" || ! -d "configs" ]]; then
    echo "ERROR: The official D-FINE repo is not cloned into this folder."
    echo ""
    echo "Run once from the repo root:"
    echo "  cd ~/traffic-vision"
    echo "  git clone https://github.com/Peterande/D-FINE.git D-FINE-tmp --depth=1"
    echo "  cp -rn D-FINE-tmp/. D-FINE/"
    echo "  rm -rf D-FINE-tmp"
    exit 1
fi

# ── Step 1: Install missing D-FINE dependencies ───────────────────────────────
echo "[1/4] Installing D-FINE dependencies..."
# Navigate to repo root so uv finds pyproject.toml
cd "$SCRIPT_DIR/.."
uv add tensorboard loguru faster-coco-eval calflops transformers --quiet
cd "$SCRIPT_DIR"
echo "  Dependencies OK"

# ── Step 2: Prepare COCO-format dataset ───────────────────────────────────────
echo ""
echo "[2/4] Preparing COCO-format dataset..."
uv run python prepare_dfine_dataset.py

# ── Step 3: Download pre-trained weights ─────────────────────────────────────
echo ""
echo "[3/4] Checking pre-trained weights..."
if [[ ! -f "$WEIGHTS" ]]; then
    echo "  Downloading D-FINE-M (obj2coco)..."
    uv run python download_dfine_weights.py --model m --pretrain obj2coco
else
    echo "  Already present: $WEIGHTS"
fi

# ── Step 4: Fine-tune ─────────────────────────────────────────────────────────
echo ""
echo "[4/4] Launching fine-tuning..."
echo ""

uv run python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_port="$PORT" \
    train.py \
    -c "$CONFIG" \
    --use-amp \
    --seed="$SEED" \
    -t "$WEIGHTS"

echo ""
echo "Done. Checkpoints saved to output/dfine_m_traffic/"
