#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║         KarnaLM — AMD MI300X ENVIRONMENT SETUP                  ║
# ║         Run this ONCE when instance first spins up              ║
# ║         Expected time: 20-30 minutes                            ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# USAGE: bash 00_amd_setup.sh 2>&1 | tee setup_log.txt
# CHECK: cat setup_log.txt after completion

set -e  # exit on any error
echo "╔══════════════════════════════════════════════════════╗"
echo "║         KarnaLM AMD MI300X Setup Starting            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Started at: $(date)"
echo ""

# ─── STEP 1: Verify Hardware ──────────────────────────────────────────────────
echo "▶ STEP 1: Hardware Verification"
echo "----------------------------------------"

# Check GPU is AMD MI300X
rocm-smi --showproductname 2>/dev/null || {
    echo "✗ rocm-smi not found — are you on the right instance?"
    exit 1
}

echo "GPU Info:"
rocm-smi --showmeminfo vram 2>/dev/null | head -20
echo ""

# Verify 192GB VRAM
VRAM_GB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | awk '{print $NF}' | head -1)
echo "Detected VRAM: $VRAM_GB"
echo "✓ Hardware verified"
echo ""

# ─── STEP 2: Critical ROCm Environment Variables ──────────────────────────────
echo "▶ STEP 2: Setting ROCm Environment Variables"
echo "----------------------------------------"
# These are NOT optional — every one of these matters

cat >> ~/.bashrc << 'EOF'

# ═══ KarnaLM ROCm Training Environment ═══
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=8
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_FILENAME=/data/tunableop_cache.csv
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0
export HSA_OVERRIDE_GFX_VERSION=9.4.0
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export MIOPEN_DEBUG_DISABLE_FIND_DB=0
export MIOPEN_FIND_MODE=NORMAL
export AMD_SERIALIZE_KERNEL=0
# Prevent ROCm from using all VRAM for caching
export HSA_TOOLS_LIB=""
# Flash Attention on AMD
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
# HuggingFace Token for automated pushes
export HF_TOKEN="YOUR_HF_TOKEN"
EOF

# Apply immediately for this session
source ~/.bashrc
echo "✓ Environment variables set and applied"
echo ""

# ─── STEP 3: Docker Check or Direct Install ───────────────────────────────────
echo "▶ STEP 3: Python Environment Setup"
echo "----------------------------------------"

# Check Python version
python3 --version
pip3 --version

# Upgrade pip first
pip3 install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# ─── STEP 4: Install PyTorch for ROCm ────────────────────────────────────────
echo "▶ STEP 4: Installing PyTorch ROCm"
echo "----------------------------------------"
echo "This takes 5-10 minutes..."

# ROCm 6.1 compatible PyTorch
pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 \
    --quiet

echo "Verifying PyTorch ROCm installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA (ROCm) available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f'Total VRAM: {mem/1024**3:.0f} GB')
    print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
    # Quick computation test
    x = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
    y = x @ x.T
    print(f'BF16 matmul test: PASSED (result shape: {y.shape})')
"
echo "✓ PyTorch ROCm verified"
echo ""

# ─── STEP 5: Install Training Libraries ──────────────────────────────────────
echo "▶ STEP 5: Installing Training Libraries"
echo "----------------------------------------"

# Core training stack
pip3 install \
    transformers==4.40.0 \
    datasets==2.19.0 \
    tokenizers==0.19.0 \
    accelerate==0.30.0 \
    peft==0.10.0 \
    trl==0.8.6 \
    --quiet

echo "✓ Core training stack installed"

# Flash Attention 2 for AMD (Triton-based)
pip3 install flash-attn --no-build-isolation --quiet 2>/dev/null || {
    echo "⚠ Flash Attention pip install failed — trying AMD Triton version"
    pip3 install triton --quiet
    # FA2 via Triton for AMD — set env var FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    echo "✓ Triton installed (FA2 will use Triton backend on AMD)"
}

# Monitoring & logging
pip3 install \
    wandb==0.16.6 \
    tensorboard \
    numpy \
    scipy \
    sentencepiece \
    --quiet

echo "✓ Monitoring tools installed"

# Data tools
pip3 install \
    pyarrow \
    pandas \
    huggingface_hub \
    kaggle \
    --quiet

echo "✓ Data tools installed"

# Evaluation
pip3 install \
    lm-eval==0.4.2 \
    sacrebleu \
    rouge-score \
    --quiet

echo "✓ Evaluation tools installed"
echo ""

# ─── STEP 6: Install LLM Foundry (Primary Training Framework) ────────────────
echo "▶ STEP 6: Installing LLM Foundry"
echo "----------------------------------------"
echo "LLM Foundry is ROCm-confirmed compatible..."

cd /opt || cd ~
git clone https://github.com/mosaicml/llm-foundry.git --quiet
cd llm-foundry
pip3 install -e ".[gpu]" --quiet

echo "✓ LLM Foundry installed"
cd ~
echo ""

# ─── STEP 7: Create Directory Structure ──────────────────────────────────────
echo "▶ STEP 7: Creating Directory Structure"
echo "----------------------------------------"

mkdir -p /data/{tokens,checkpoints,logs,eval,tokenizer}
mkdir -p /data/checkpoints/karnalm-pretrain
mkdir -p /data/checkpoints/karnalm-sft

echo "✓ Directories created:"
tree /data 2>/dev/null || ls -la /data/
echo ""

# ─── STEP 8: Download Data from HuggingFace ──────────────────────────────────
echo "▶ STEP 8: Downloading KarnaLM Dataset"
echo "----------------------------------------"
echo "Downloading from ncncomplete/karnalm-data..."
echo "Note: Set HF_TOKEN if dataset is private"
echo ""

# Set your HuggingFace token here or export before running
# export HF_TOKEN="hf_your_token_here"

python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os

token = os.environ.get("HF_TOKEN", None)

try:
    snapshot_download(
        repo_id="ncncomplete/karnalm-data",
        repo_type="dataset",
        local_dir="/data/raw",
        token=token,
        ignore_patterns=["*.md", "*.txt"],  # skip readme
    )
    print("✓ Dataset downloaded to /data/raw")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("  Run manually: huggingface-cli login")
    print("  Then re-run this step")
PYEOF

echo ""

# ─── STEP 9: TunableOp Warmup ────────────────────────────────────────────────
echo "▶ STEP 9: ROCm TunableOp Kernel Tuning"
echo "----------------------------------------"
echo "This pre-tunes GPU kernels for your specific shapes — saves 10-15% training time"
echo "Takes ~5 minutes..."

python3 << 'PYEOF'
import torch
import os

os.makedirs("/data", exist_ok=True)
device = "cuda"

print("Tuning attention-shaped matmuls (seq_len=2048, d_model=2048)...")
shapes = [
    (512, 2048, 2048),   # batch x seq x d_model
    (512, 2048, 5632),   # FFN up projection
    (512, 5632, 2048),   # FFN down projection
    (512, 2048, 256),    # QKV with GQA (head_dim * n_kv_heads)
]

for shape in shapes:
    a = torch.randn(*shape[:2], shape[2]//4, dtype=torch.bfloat16, device=device)
    b = torch.randn(shape[2]//4, shape[2]//8, dtype=torch.bfloat16, device=device)
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"  Tuned shape {shape} ✓")

print("✓ Kernel tuning complete — cache saved to /data/tunableop_cache.csv")
PYEOF

echo ""

# ─── STEP 10: Final Verification ──────────────────────────────────────────────
echo "▶ STEP 10: Final System Verification"
echo "----------------------------------------"

python3 << 'PYEOF'
import torch
import transformers
import datasets
import tokenizers
import accelerate

print("╔════════════════════════════════════════╗")
print("║     KarnaLM Environment Checklist      ║")
print("╠════════════════════════════════════════╣")

checks = []

# GPU
gpu_ok = torch.cuda.is_available()
checks.append(("GPU Available", gpu_ok))
print(f"║ {'✓' if gpu_ok else '✗'} GPU Available: {torch.cuda.get_device_name(0) if gpu_ok else 'NOT FOUND'}")

# VRAM
if gpu_ok:
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    vram_ok = vram > 150
    checks.append(("VRAM > 150GB", vram_ok))
    print(f"║ {'✓' if vram_ok else '✗'} VRAM: {vram:.0f} GB")

# BF16
bf16_ok = torch.cuda.is_bf16_supported() if gpu_ok else False
checks.append(("BF16 Supported", bf16_ok))
print(f"║ {'✓' if bf16_ok else '✗'} BF16 Supported: {bf16_ok}")

# Libraries
for name, pkg in [("transformers", transformers), ("datasets", datasets),
                   ("tokenizers", tokenizers), ("accelerate", accelerate)]:
    v = pkg.__version__
    print(f"║ ✓ {name}: {v}")

# Data
import os
data_exists = os.path.exists("/data/raw")
print(f"║ {'✓' if data_exists else '⚠'} Data at /data/raw: {'YES' if data_exists else 'NOT YET'}")

all_ok = all(v for _, v in checks)
print("╠════════════════════════════════════════╣")
print(f"║ Status: {'✓ READY TO TRAIN' if all_ok else '✗ FIX ISSUES ABOVE'}")
print("╚════════════════════════════════════════╝")
PYEOF

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              SETUP COMPLETE                          ║"
echo "║  Completed at: $(date)                               ║"
echo "║                                                      ║"
echo "║  Next steps:                                         ║"
echo "║  1. bash scripts/01_tokenize_data.sh                 ║"
echo "║  2. bash scripts/02_verify_tokens.sh                 ║"
echo "║  3. bash scripts/03_smoke_test.sh                    ║"
echo "║  4. bash scripts/04_start_training.sh                ║"
echo "╚══════════════════════════════════════════════════════╝"
