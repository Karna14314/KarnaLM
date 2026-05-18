#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║     KarnaLM — monitor.sh                                        ║
# ║     Run: watch -n 30 bash scripts/monitor.sh                   ║
# ╚══════════════════════════════════════════════════════════════════╝

echo "═══════════════════════════════════════════════"
echo "   KarnaLM Training Monitor — $(date '+%H:%M:%S')"
echo "═══════════════════════════════════════════════"

echo ""
echo "▶ GPU Status:"
rocm-smi --showuse --showmemuse 2>/dev/null | grep -A5 "GPU\[0\]" || \
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader 2>/dev/null

echo ""
echo "▶ Last 5 training lines:"
tail -5 /data/logs/training.log 2>/dev/null || echo "  Log not found yet"

echo ""
echo "▶ Latest checkpoint:"
ls -lt /data/checkpoints/karnalm-pretrain/*.pt 2>/dev/null | head -3 || \
    echo "  No checkpoints yet"

echo ""
echo "▶ Disk usage:"
df -h /data 2>/dev/null | tail -1

echo ""
echo "▶ Process running:"
pgrep -a python 2>/dev/null | grep pretrain || echo "  Training not running"
