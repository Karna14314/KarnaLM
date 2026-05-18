#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  KarnaLM — Verify Environment & Launch Training                 ║
# ║  This script runs all pre-flight checks and auto-launches       ║
# ║  training in a tmux session if everything passes.               ║
# ╚══════════════════════════════════════════════════════════════════╝

set -e

echo "═══════════════════════════════════════════════════"
echo "  KarnaLM Pre-Flight Verification & Launch"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── Step 1: Source environment ───────────────────────────────────────────────
echo "▶ Step 1: Loading environment..."
source ~/.bashrc
source /root/karnalm_venv/bin/activate
echo "  ✓ Environment loaded"
echo ""

# ─── Step 2: Check GPU VRAM ──────────────────────────────────────────────────
echo "▶ Step 2: GPU VRAM Status"
rocm-smi --showmeminfo vram
echo ""

# ─── Step 3: Quick GPU test ──────────────────────────────────────────────────
echo "▶ Step 3: Quick GPU Matmul Test"
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
result = (x @ x.T).shape
print(f'  Matmul result: {result}')
print('  ✓ GPU OK')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "  ✗ GPU TEST FAILED — Driver may be wedged."
    echo "  Action: Restart the instance and try again."
    exit 1
fi
echo ""

# ─── Step 4: Run Smoke Test ──────────────────────────────────────────────────
echo "▶ Step 4: Running Full Smoke Test..."
echo "  (This will take 5-10 minutes for the 1.1B benchmark)"
echo ""

python3 training_scripts/02_smoke_test.py
SMOKE_EXIT=$?

if [ $SMOKE_EXIT -ne 0 ]; then
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  ✗ SMOKE TEST FAILED ($SMOKE_EXIT checks failed)"
    echo "  Fix the issues above before launching training."
    echo "═══════════════════════════════════════════════════"
    exit $SMOKE_EXIT
fi

# ─── Step 5: Launch Training in tmux ─────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ ALL CHECKS PASSED — Launching Training"
echo "═══════════════════════════════════════════════════"
echo ""

# Create log directory
mkdir -p /data/logs

# Kill any existing karnalm tmux session
tmux kill-session -t karnalm 2>/dev/null || true

# Launch training in a new tmux session
tmux new-session -d -s karnalm \
    "source ~/.bashrc && source /root/karnalm_venv/bin/activate && python3 training_scripts/03_pretrain.py 2>&1 | tee /data/logs/training.log"

echo "  Training launched in tmux session 'karnalm'"
echo ""
echo "  Monitor with:  tail -f /data/logs/training.log"
echo "  Attach with:   tmux attach -t karnalm"
echo "  Detach with:   Ctrl+B, then D"
echo ""
echo "═══════════════════════════════════════════════════"
echo "  KarnaLM training is now running!"
echo "═══════════════════════════════════════════════════"
