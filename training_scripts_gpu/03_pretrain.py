#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     KarnaLM — SCRIPT 03: MAIN PRETRAINING                       ║
║     1.1B LLaMA-style model on AMD MI300X (192GB)                ║
║     Target: 15B tokens in ~45 hours                             ║
║                                                                  ║
║     BEFORE RUNNING:                                              ║
║     1. bash scripts/00_amd_setup.sh         (done once)         ║
║     2. python3 scripts/01_tokenize_data.py  (done once)         ║
║     3. python3 scripts/02_smoke_test.py     (must PASS)         ║
║                                                                  ║
║     RUN: python3 scripts/03_pretrain.py                         ║
║     RESUME: python3 scripts/03_pretrain.py --resume             ║
║                                                                  ║
║     MONITOR: tail -f /data/logs/training.log                    ║
║              watch -n 30 scripts/monitor.sh                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, math, json, logging, argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaConfig, LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
import threading
import glob
import random
import wandb
from huggingface_hub import HfApi, login

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.pop("TORCH_ATTENTION_BACKEND", None)
os.environ["WANDB_DISABLED"] = "true"

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.pop("TORCH_ATTENTION_BACKEND", None)
# ─── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
parser.add_argument("--config", type=str, default=None, help="Path to config JSON override")
args = parser.parse_args()

# ─── LOGGING ───────────────────────────────────────────────────────────────────
os.makedirs("/data/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/data/logs/training.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Paths
    train_data_dir:  str = "/mnt/scratch/shards"
    ckpt_dir:        str = "/data/checkpoints/karnalm-360m"
    log_dir:         str = "/data/logs"
    tok_path:        str = "/mnt/scratch/tokenizer/karnalm_tokenizer.json" 
    hf_repo:         str = "ncncomplete/KarnaLM-360M-base"

    # Model — 360M LLaMA architecture
    vocab_size:        int   = 52_000
    hidden_size:       int   = 1024
    num_layers:        int   = 24
    num_heads:         int   = 16
    num_kv_heads:      int   = 8
    intermediate_size: int   = 2816
    max_seq_len:       int   = 2048
    rope_theta:        float = 10000.0

    # Training
    batch_size:        int   = 16
    grad_accum:        int   = 4
    total_tokens:      int   = 12_800_000_000
    max_lr:            float = 5e-4
    min_lr:            float = 5e-5
    warmup_steps:      int   = 2000
    weight_decay:      float = 0.1
    grad_clip:         float = 1.0
    beta1:             float = 0.9
    beta2:             float = 0.95
    eps:               float = 1e-8
    gradient_checkpointing: bool = False
    dtype:             str   = "bfloat16"

    # Checkpointing
    save_every:   int = 2000
    keep_ckpts:   int = 3
    val_every:    int = 500
    log_every:    int = 10
    push_every:   int = 10000 

    # Derived — computed in __post_init__
    tokens_per_step: int = 0
    total_steps:     int = 0

    def __post_init__(self):
        self.tokens_per_step = self.batch_size * self.max_seq_len * self.grad_accum
        self.total_steps = self.total_tokens // self.tokens_per_step

CFG = TrainingConfig()

# Load override config if provided
if args.config:
    with open(args.config) as f:
        overrides = json.load(f)
    for k, v in overrides.items():
        setattr(CFG, k, v)
    CFG.__post_init__()

log.info("=" * 60)
log.info("KarnaLM Pretraining Configuration")
log.info("=" * 60)
log.info(f"Model: {CFG.hidden_size}d, {CFG.num_layers}L, {CFG.num_heads}H (GQA: {CFG.num_kv_heads} KV heads)")
log.info(f"Sequence length: {CFG.max_seq_len}")
log.info(f"Batch: {CFG.batch_size} seqs × {CFG.grad_accum} accum = {CFG.tokens_per_step/1e3:.0f}K tokens/step")
log.info(f"Total steps: {CFG.total_steps:,} ({CFG.total_tokens/1e9:.0f}B tokens)")
log.info(f"LR: {CFG.max_lr} → {CFG.min_lr} (warmup {CFG.warmup_steps} steps)")
log.info("=" * 60)

os.makedirs(CFG.ckpt_dir, exist_ok=True)

if os.environ.get("HF_TOKEN"):
    login(token=os.environ.get("HF_TOKEN"))
    log.info(f"Logged into Hugging Face. Checkpoints will be pushed to {CFG.hf_repo}")
else:
    log.warning("No HF_TOKEN found. Automatic pushing to Hugging Face will be skipped.")

# ─── DATASET ───────────────────────────────────────────────────────────────────

class ShardedBinaryDataset:
    """Memory-mapped sharded binary dataset for MI300X"""
    def __init__(self, shard_pattern: str, seq_len: int, split='train'):
        import glob
        import random
        self.shard_files = sorted(glob.glob(shard_pattern))
        random.shuffle(self.shard_files)
        
        if not self.shard_files:
            log.error(f"No shards found for pattern: {shard_pattern}")
            raise FileNotFoundError(f"No shards found for pattern: {shard_pattern}")
        self.seq_len = seq_len
        self.split = split
        
        # Pre-load shard sizes and open memmaps persistently
        from pathlib import Path
        self.shard_sizes = [Path(f).stat().st_size // 2 for f in self.shard_files]
        self.mmaps = [np.memmap(f, dtype=np.uint16, mode='r') for f in self.shard_files]
        
        total_tokens = sum(self.shard_sizes)
        log.info(f"Dataset ({split}): {len(self.shard_files)} shards, {total_tokens/1e9:.2f}B tokens")

    def get_batch(self, batch_size):
        # 1. Select shard
        shard_idx = random.randint(0, len(self.shard_files) - 1)
        data = self.mmaps[shard_idx]
        
        # 2. Sample
        ix = torch.randint(len(data) - self.seq_len - 1, (batch_size,))
        
        x = torch.stack([torch.from_numpy((data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
        
        return x, y

train_dataset = ShardedBinaryDataset(f"{CFG.train_data_dir}/train_shard_*.bin", CFG.max_seq_len, split='train')
val_dataset   = ShardedBinaryDataset(f"{CFG.train_data_dir}/val.bin",     CFG.max_seq_len, split='val')

# ─── MODEL ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")

model_config = LlamaConfig(
    vocab_size=CFG.vocab_size,
    hidden_size=CFG.hidden_size,
    num_hidden_layers=CFG.num_layers,
    num_attention_heads=CFG.num_heads,
    num_key_value_heads=CFG.num_kv_heads,
    intermediate_size=CFG.intermediate_size,
    max_position_embeddings=CFG.max_seq_len,
    rope_theta=CFG.rope_theta,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    attn_implementation="sdpa",
    use_cache=False,
)

log.info("Initializing model...")
model = LlamaForCausalLM(model_config)

if CFG.gradient_checkpointing:
    log.info("Enabling gradient checkpointing for VRAM safety...")
    model.gradient_checkpointing_enable()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
log.info(f"Model: {total_params/1e6:.0f}M parameters")
log.info(f"Architecture: {CFG.num_layers}L {CFG.hidden_size}d {CFG.num_heads}H")
log.info(f"Training: {CFG.total_tokens/1e9:.1f}B tokens, {CFG.total_steps:,} steps")
log.info(f"Batch: {CFG.batch_size} seqs × {CFG.grad_accum} accum = {CFG.tokens_per_step/1e3:.0f}K tokens/step")
log.info(f"LR: {CFG.max_lr} → {CFG.min_lr} over {CFG.total_steps:,} steps ({CFG.warmup_steps} warmup)")

model = model.to(device, dtype=torch.bfloat16)
torch.cuda.synchronize()

vram_after_model = torch.cuda.memory_allocated() / 1024**3
log.info(f"VRAM after model init: {vram_after_model:.1f} GB")

# ─── OPTIMIZER ─────────────────────────────────────────────────────────────────

# Separate weight decay: apply to weight matrices, not biases/norms
decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params,    "weight_decay": CFG.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=CFG.max_lr,
    betas=(CFG.beta1, CFG.beta2),
    eps=CFG.eps,
    fused=True,   # fused AdamW — faster on CUDA/ROCm
)

# Cosine LR schedule with linear warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CFG.warmup_steps,
    num_training_steps=CFG.total_steps,
    last_epoch=-1,
)

# ─── CHECKPOINT ────────────────────────────────────────────────────────────────

def push_to_hub(step: int, val_loss: float):
    try:
        from huggingface_hub import HfApi
        from transformers import PreTrainedTokenizerFast
        
        hf_path = f"{CFG.ckpt_dir}/hf_checkpoint"
        os.makedirs(hf_path, exist_ok=True)
        
        model.save_pretrained(hf_path)
        
        tokenizer_hf = PreTrainedTokenizerFast(
            tokenizer_file=CFG.tok_path,
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
        tokenizer_hf.save_pretrained(hf_path)
        
        api = HfApi()
        api.upload_folder(
            folder_path=hf_path,
            repo_id=CFG.hf_repo,
            repo_type="model",
            commit_message=f"step {step} | val_loss {val_loss:.4f} | {step*CFG.tokens_per_step/1e9:.1f}B tokens",
            ignore_patterns=["*.pt"],
        )
        log.info(f"✓ Pushed to Hugging Face: {CFG.hf_repo} at step {step}")
    except Exception as e:
        log.warning(f"HF push failed (non-fatal, training continues): {e}")

def save_checkpoint(step, accum_loss, val_loss):
    ckpt_path = Path(CFG.ckpt_dir) / f"step_{step:07d}.pt"
    
    # Save standard PyTorch checkpoint
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": accum_loss,
        "val_loss": val_loss,
    }, ckpt_path)
    log.info(f"Saved local checkpoint: {ckpt_path.name}")

    # Keep only last N checkpoints
    ckpts = sorted(Path(CFG.ckpt_dir).glob("step_*.pt"))
    while len(ckpts) > CFG.keep_ckpts:
        old_pt = ckpts[0]
        old_pt.unlink()
        log.info(f"Deleted old checkpoint: {old_pt}")
        ckpts = ckpts[1:]

def load_checkpoint():
    ckpts = sorted(Path(CFG.ckpt_dir).glob("step_*.pt"))
    if not ckpts:
        log.info("No checkpoint found — starting from scratch")
        return 0
    latest = ckpts[-1]
    log.info(f"Resuming from: {latest}")
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_step = ckpt["step"]
    log.info(f"Resumed from step {start_step:,}")
    return start_step

start_step = load_checkpoint()

# ─── VALIDATION ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for i in range(50):  # 50 batches = ~3M tokens validation
        x, y = val_dataset.get_batch(CFG.batch_size)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        losses.append(out.loss.item())
    model.train()
    mean_loss = np.mean(losses)
    perplexity = math.exp(min(mean_loss, 20))  # cap at exp(20) to avoid overflow
    return mean_loss, perplexity

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────

log.info("\n" + "=" * 60)
log.info("TRAINING START")
log.info(f"Start step: {start_step:,} / {CFG.total_steps:,}")
log.info(f"Tokens already trained: {start_step * CFG.tokens_per_step / 1e9:.2f}B")
log.info("=" * 60)

model.train()
optimizer.zero_grad()

step = start_step
tokens_trained = start_step * CFG.tokens_per_step
t0 = time.time()
t_step_start = time.time()

# Metrics tracking
running_loss = 0.0
running_grad_norm = 0.0
# Initialize WandB if not disabled
if os.environ.get("WANDB_DISABLED") != "true":
    wandb.init(
        project="KarnaLM-360M",
        config=asdict(CFG),
    )

while step < CFG.total_steps:
    # ── Gradient accumulation loop ──────────────────────────────────────────
    accum_loss = 0.0

    for micro_step in range(CFG.grad_accum):
        x, y = train_dataset.get_batch(CFG.batch_size)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
            loss = out.loss / CFG.grad_accum  # scale loss for accumulation

        loss.backward()
        accum_loss += loss.item()

    # ── Gradient clip & optimizer step ──────────────────────────────────────
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory

    step += 1
    tokens_trained += CFG.tokens_per_step
    running_loss     += accum_loss
    running_grad_norm += grad_norm.item()

    # ── Logging ──────────────────────────────────────────────────────────────
    if step % CFG.log_every == 0:
        t_now = time.time()
        dt = t_now - t_step_start
        t_step_start = t_now
        elapsed_h = (t_now - t0) / 3600

        tokens_per_sec = (CFG.tokens_per_step * CFG.log_every) / dt
        remaining_tokens = CFG.total_tokens - tokens_trained
        eta_h = remaining_tokens / tokens_per_sec / 3600

        avg_loss = running_loss / CFG.log_every
        avg_grad = running_grad_norm / CFG.log_every
        current_lr = scheduler.get_last_lr()[0]
        vram_used = torch.cuda.memory_allocated() / 1024**3

        log.info(
            f"step={step:>6}/{CFG.total_steps} | "
            f"loss={avg_loss:.4f} | "
            f"ppl={math.exp(min(avg_loss,20)):.1f} | "
            f"grad={avg_grad:.3f} | "
            f"lr={current_lr:.2e} | "
            f"tok={tokens_trained/1e9:.2f}B | "
            f"tok/s={tokens_per_sec:,.0f} | "
            f"vram={vram_used:.0f}GB | "
            f"elapsed={elapsed_h:.1f}h | "
            f"eta={eta_h:.1f}h"
        )

        # Log to WandB
        if os.environ.get("WANDB_DISABLED") != "true":
            wandb.log({
                "train/loss": avg_loss,
                "train/ppl": math.exp(min(avg_loss, 20)),
                "train/grad_norm": avg_grad,
                "train/lr": current_lr,
                "train/tokens_B": tokens_trained / 1e9,
                "train/tokens_per_sec": tokens_per_sec,
                "system/vram_gb": vram_used,
            }, step=step)

        # Loss milestone warnings
        if step == 5000 and avg_loss > 6.0:
            log.warning(f"⚠ Loss {avg_loss:.3f} too high at step 5000 — expected <6.0")
        if step == 10000 and avg_loss > 4.5:
            log.warning(f"⚠ Loss {avg_loss:.3f} too high at step 10000 — expected <4.5")
        if step == 20000 and avg_loss > 3.8:
            log.warning(f"⚠ Loss {avg_loss:.3f} too high at step 20000 — expected <3.8")
        
        # ETA recalculation every 100 steps using real throughput
        if step % 100 == 0:
            tokens_remaining = CFG.total_tokens - tokens_trained
            real_eta_h = tokens_remaining / tokens_per_sec / 3600
            log.info(f"  → Real ETA: {real_eta_h:.1f}h at current {tokens_per_sec:,.0f} tok/s")

        running_loss = 0.0
        running_grad_norm = 0.0

    # ── Validation ───────────────────────────────────────────────────────────
    if step % CFG.val_every == 0:
        val_loss, val_ppl = evaluate()
        log.info(f"  ▶ VAL step={step}: loss={val_loss:.4f}, ppl={val_ppl:.1f}")
        if os.environ.get("WANDB_DISABLED") != "true":
            wandb.log({
                "val/loss": val_loss,
                "val/ppl": val_ppl,
            }, step=step)

        # Early warning: val loss diverging from train
        if step > 2000:
            recent_train = accum_loss  # approximate
            if val_loss > recent_train * 1.3:
                log.warning("⚠ Val loss >> train loss — possible overfitting or data issue")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if step % CFG.save_every == 0:
        val_loss, _ = evaluate()
        save_checkpoint(step, accum_loss, val_loss)
        
    if step % CFG.push_every == 0:
        val_loss, _ = evaluate()
        push_to_hub(step, val_loss)

# ─── TRAINING COMPLETE ────────────────────────────────────────────────────────

total_time_h = (time.time() - t0) / 3600
log.info("\n" + "=" * 60)
log.info("TRAINING COMPLETE")
log.info(f"Total time: {total_time_h:.1f} hours")
log.info(f"Total tokens: {tokens_trained/1e9:.2f}B")
log.info(f"Final avg tok/sec: {tokens_trained / (total_time_h * 3600):,.0f}")
log.info("=" * 60)

log.info("Training complete — running final evaluation and push")
final_val_loss, final_ppl = evaluate()
log.info(f"Final val loss: {final_val_loss:.4f} | ppl: {final_ppl:.1f}")

# Save final HF model
final_hf_path = f"{CFG.ckpt_dir}/final"
model.save_pretrained(final_hf_path)
from transformers import PreTrainedTokenizerFast
tokenizer_hf = PreTrainedTokenizerFast(
    tokenizer_file=CFG.tok_path,
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
    unk_token="<|unk|>",
)
tokenizer_hf.save_pretrained(final_hf_path)

log.info(f"Model saved locally: {final_hf_path}")
log.info("Pushing final model to HuggingFace...")
push_to_hub(CFG.total_steps, final_val_loss)
log.info("Done. Run 04_sft.py next.")
