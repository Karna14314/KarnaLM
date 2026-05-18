#!/bin/bash
set -e  # exit on any error

# ─── Environment ──────────────────────────────────────────────────
source ~/.bashrc
source /root/karnalm_venv/bin/activate

LOG=/data/logs/pipeline.log
mkdir -p /data/logs /data/checkpoints/karnalm-360m

exec > >(tee -a "$LOG") 2>&1

echo "═══════════════════════════════════════════════════"
echo "  KarnaLM Full Pipeline — $(date)"
echo "═══════════════════════════════════════════════════"

# ─── Step 1: Pretrain ─────────────────────────────────────────────
PRETRAIN_DONE="/data/checkpoints/karnalm-360m/final/config.json"

if [ -f "$PRETRAIN_DONE" ]; then
    echo "✓ Pretrain already complete — skipping to SFT"
else
    echo "▶ Starting pretraining..."
    python3 training_scripts/03_pretrain.py
    
    if [ ! -f "$PRETRAIN_DONE" ]; then
        echo "✗ Pretrain failed — config.json not found at $PRETRAIN_DONE"
        exit 1
    fi
    echo "✓ Pretraining complete at $(date)"
fi

# ─── Step 2: Push base model to HuggingFace ───────────────────────
echo "▶ Pushing base model to HuggingFace..."

python3 - << 'PYEOF'
import sys
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerFast
import os

MODEL_PATH = "/data/checkpoints/karnalm-360m/final"
TOK_PATH   = "/mnt/scratch/shards/tokenizer.json"
HF_REPO    = "ncncomplete/KarnaLM-360M-base"

try:
    # Save tokenizer alongside model if not already there
    if not os.path.exists(f"{MODEL_PATH}/tokenizer.json"):
        tok = PreTrainedTokenizerFast(
            tokenizer_file=TOK_PATH,
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
        tok.save_pretrained(MODEL_PATH)
        print("✓ Tokenizer saved alongside model")

    api = HfApi()
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="KarnaLM-360M-base: full pretrain complete",
        ignore_patterns=["*.pt", "*.bin.index*"],
    )
    print(f"✓ Base model pushed to {HF_REPO}")
except Exception as e:
    print(f"✗ HF push failed: {e}")
    print("  Continuing to SFT anyway — can push manually later")
PYEOF

# ─── Step 3: SFT ──────────────────────────────────────────────────
SFT_DONE="/data/checkpoints/karnalm-360m-chat/final/config.json"

if [ -f "$SFT_DONE" ]; then
    echo "✓ SFT already complete — skipping to final push"
else
    echo "▶ Starting SFT at $(date)..."
    python3 training_scripts/04_sft.py
    
    if [ ! -f "$SFT_DONE" ]; then
        echo "✗ SFT failed — config.json not found"
        exit 1
    fi
    echo "✓ SFT complete at $(date)"
fi

# ─── Step 4: Push chat model to HuggingFace ───────────────────────
echo "▶ Pushing chat model to HuggingFace..."

python3 - << 'PYEOF'
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerFast
import os

MODEL_PATH = "/data/checkpoints/karnalm-360m-chat/final"
TOK_PATH   = "/mnt/scratch/shards/tokenizer.json"
HF_REPO    = "ncncomplete/KarnaLM-360M-chat"

try:
    if not os.path.exists(f"{MODEL_PATH}/tokenizer.json"):
        tok = PreTrainedTokenizerFast(
            tokenizer_file=TOK_PATH,
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
        tok.save_pretrained(MODEL_PATH)

    api = HfApi()
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="KarnaLM-360M-chat: SFT complete on Aya EN+HI+TE",
        ignore_patterns=["*.pt"],
    )
    print(f"✓ Chat model pushed to {HF_REPO}")
except Exception as e:
    print(f"✗ HF push failed: {e}")
PYEOF

# ─── Step 5: Run benchmarks ───────────────────────────────────────
echo "▶ Running basic evaluation..."

python3 - << 'PYEOF'
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

MODEL_PATH = "/data/checkpoints/karnalm-360m-chat/final"
TOK_PATH   = "/mnt/scratch/shards/tokenizer.json"

print("Loading model for inference check...")
tok = PreTrainedTokenizerFast(
    tokenizer_file=TOK_PATH,
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
    unk_token="<|unk|>",
)
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

prompts = [
    "The capital of India is",
    "भारत की राजधानी",
    "తెలుగు భాష",
    "Hyderabad is a city in",
]

print("\n─── Generation Samples ───")
for p in prompts:
    inputs = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
    generated = tok.decode(out[0], skip_special_tokens=True)
    print(f"  IN:  {p}")
    print(f"  OUT: {generated}")
    print()

print("✓ Inference check complete")
PYEOF

# ─── Done ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pipeline complete at $(date)"
echo "  Base model: https://huggingface.co/ncncomplete/KarnaLM-360M-base"
echo "  Chat model: https://huggingface.co/ncncomplete/KarnaLM-360M-chat"
echo "  Full log:   /data/logs/pipeline.log"
echo "═══════════════════════════════════════════════════"
