#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_eval.sh  —  Evaluate a D-FINE-M checkpoint on val and test splits.
#
# Usage (from D-FINE/ directory):
#   bash run_eval.sh                                    # uses best.pth from default output dir
#   bash run_eval.sh output/dfine_m_traffic/best.pth    # explicit checkpoint
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG="custom/configs/dfine_m_traffic.yml"
NPROC=1
PORT=7778   # different from training port to avoid collision

# Accept checkpoint path as the first positional argument, or use a default
CHECKPOINT="${1:-output/dfine_m_traffic/best.pth}"

# ── Resolve script directory ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==========================================="
echo " D-FINE-M  Traffic Detection  —  Evaluation"
echo "==========================================="
echo "Config     : $CONFIG"
echo "Checkpoint : $CHECKPOINT"
echo "Date       : $(date)"
echo ""

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Make sure training has completed, or pass a path as the first argument:"
    echo "  bash run_eval.sh path/to/checkpoint.pth"
    exit 1
fi

# ── Evaluate on val split ─────────────────────────────────────────────────────
echo "── Val split ──────────────────────────────────────────────────────────"
uv run python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_port="$PORT" \
    train.py \
    -c "$CONFIG" \
    --test-only \
    -r "$CHECKPOINT"

echo ""

# ── Evaluate on test split ────────────────────────────────────────────────────
echo "── Test split ─────────────────────────────────────────────────────────"
uv run python eval_dfine.py \
    --checkpoint "$CHECKPOINT" \
    --config     "$CONFIG" \
    --split      test \
    --nproc      "$NPROC"

echo ""
echo "Evaluation complete."
