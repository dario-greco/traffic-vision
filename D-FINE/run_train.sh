#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_train.sh  —  Fine-tune D-FINE-M on the traffic detection dataset.
#
# Run from the D-FINE/ directory:
#   bash run_train.sh
#
# Or submit to a Slurm cluster (see submit_dfine.sh for a ready-made template).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG="custom/configs/dfine_m_traffic.yml"
WEIGHTS="custom/weights/dfine_m_obj2coco.pth"
NPROC=1          # set to the number of GPUs on your node
PORT=7777
SEED=0

# ── Resolve script directory (works even when called from outside D-FINE/) ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " D-FINE-M  Traffic Detection  —  Training"
echo "=========================================="
echo "Config   : $CONFIG"
echo "Weights  : $WEIGHTS"
echo "GPUs     : $NPROC"
echo "Date     : $(date)"
echo ""

# ── Step 0: Verify D-FINE repo is present ────────────────────────────────────
if [[ ! -f "train.py" ]]; then
    echo "ERROR: train.py not found. Clone the D-FINE repo first:"
    echo "  cd D-FINE/"
    echo "  git clone https://github.com/Peterande/D-FINE.git . --depth=1"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# ── Step 1: Prepare dataset (idempotent – skips if already done) ──────────────
echo "[1/3] Preparing COCO-format dataset..."
uv run python prepare_dfine_dataset.py

# ── Step 2: Download pre-trained weights if missing ──────────────────────────
echo ""
echo "[2/3] Checking pre-trained weights..."
if [[ ! -f "$WEIGHTS" ]]; then
    echo "  Weights not found — downloading D-FINE-M (obj2coco)..."
    uv run python download_dfine_weights.py --model m --pretrain obj2coco
else
    echo "  Weights found: $WEIGHTS"
fi

# ── Step 3: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Launching training..."
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
echo "Training finished. Checkpoints saved to output/dfine_m_traffic/"
