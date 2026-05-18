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
from huggingface_hub import HfApi, login

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
    train_bin:   str = "/data/tokens/train.bin"
    val_bin:     str = "/data/tokens/val.bin"
    ckpt_dir:    str = "/data/checkpoints/karnalm-pretrain"
    auto_detect_tokens: bool = True  # auto-detect total tokens from train.bin
    log_dir:     str = "/data/logs"
    tok_path:    str = "/data/tokenizer/karnalm_tokenizer.json"

    # Model architecture
    vocab_size:       int   = 52_000
    hidden_size:      int   = 2048
    num_layers:       int   = 22
    num_heads:        int   = 32
    num_kv_heads:     int   = 8        # GQA — 8 KV heads, 32 query heads
    intermediate_size: int  = 5632     # SwiGLU: 2/3 * 4 * hidden_size, round to 64
    max_seq_len:      int   = 2048
    rope_theta:       float = 10000.0

    # Training
    batch_size:       int   = 32       # safe ceiling for 192GB VRAM with gradient checkpointing
    grad_accum:       int   = 4        # effective batch = 32 * 4 = 128 sequences = 262K tokens
    max_lr:           float = 3e-4     # peak learning rate
    min_lr:           float = 3e-5     # 10% of max_lr (cosine decay floor)
    warmup_steps:     int   = 2000     # ~500M tokens warmup
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    beta1:            float = 0.9
    beta2:            float = 0.95
    eps:              float = 1e-8
    dtype:            str   = "bfloat16"
    gradient_checkpointing: bool = True # saves VRAM at the cost of slight compute overhead

    # Data & HF
    total_tokens:     int   = 0              # auto-detected from train.bin (set to 0 for auto)
    val_every:        int   = 500             # eval every N steps
    log_every:        int   = 10             # log every N steps
    save_every:       int   = 1000           # checkpoint every N steps
    keep_ckpts:       int   = 3             # keep last N checkpoints
    hf_repo_id:       str   = "KarnaDigital/KarnaLM-1.1B-chat" # auto-push repo
    hf_token:         str   = None           # HF write token

    # Derived (computed below)
    tokens_per_step:  int   = 0
    total_steps:      int   = 0

    def __post_init__(self):
        self.tokens_per_step = self.batch_size * self.max_seq_len * self.grad_accum
        # Auto-detect total tokens from train.bin if not set
        if self.auto_detect_tokens and self.total_tokens == 0:
            import numpy as np
            from pathlib import Path
            train_path = Path(self.train_bin)
            if train_path.exists():
                file_size = train_path.stat().st_size
                self.total_tokens = file_size // 2  # uint16 = 2 bytes per token
            else:
                self.total_tokens = 15_000_000_000  # fallback estimate
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

log.info("=" * 60)

os.makedirs(CFG.ckpt_dir, exist_ok=True)

if CFG.hf_token or os.environ.get("HF_TOKEN"):
    login(token=CFG.hf_token or os.environ.get("HF_TOKEN"))
    log.info(f"Logged into Hugging Face. Checkpoints will be pushed to {CFG.hf_repo_id}")
else:
    log.warning("No HF_TOKEN found. Automatic pushing to Hugging Face will be skipped.")

# ─── DATASET ───────────────────────────────────────────────────────────────────

class BinaryTokenDataset(Dataset):
    """Memory-mapped binary token dataset — zero RAM overhead, fast random access"""
    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_sequences = len(self.data) // (seq_len + 1)
        log.info(f"Dataset: {len(self.data)/1e9:.2f}B tokens → {self.n_sequences:,} sequences")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # Input: tokens[start:start+seq_len]
        # Label: tokens[start+1:start+seq_len+1] (next-token prediction)
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

train_dataset = BinaryTokenDataset(CFG.train_bin, CFG.max_seq_len)
val_dataset   = BinaryTokenDataset(CFG.val_bin,   CFG.max_seq_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

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
    use_cache=False,           # disable KV cache during training
    attn_implementation="flash_attention_2",  # FA2 via Triton on AMD
    torch_dtype=CFG.dtype,
)

log.info("Initializing model...")
model = LlamaForCausalLM(model_config)

if CFG.gradient_checkpointing:
    log.info("Enabling gradient checkpointing for VRAM safety...")
    model.gradient_checkpointing_enable()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Parameters: {total_params/1e9:.3f}B total, {trainable_params/1e9:.3f}B trainable")

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

def async_push_to_hf(ckpt_path, step):
    if not CFG.hf_repo_id:
        return
    def _push():
        try:
            api = HfApi()
            # Push the specific .pt file
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"checkpoints/step_{step:07d}.pt",
                repo_id=CFG.hf_repo_id,
                repo_type="model"
            )
            log.info(f"✓ Async HF upload complete for step {step}")
        except Exception as e:
            log.warning(f"⚠ Async HF upload failed: {e}")
    
    t = threading.Thread(target=_push)
    t.start()

def save_checkpoint(step, loss, val_loss=None):
    ckpt_path = Path(CFG.ckpt_dir) / f"step_{step:07d}.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "val_loss": val_loss,
        "config": asdict(CFG),
    }, ckpt_path)
    log.info(f"Checkpoint saved: {ckpt_path}")

    # Async push to Hugging Face
    async_push_to_hf(ckpt_path, step)

    # Keep only last N checkpoints
    ckpts = sorted(Path(CFG.ckpt_dir).glob("step_*.pt"))
    while len(ckpts) > CFG.keep_ckpts:
        ckpts[0].unlink()
        log.info(f"Deleted old checkpoint: {ckpts[0]}")
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

start_step = load_checkpoint() if args.resume else 0

# ─── VALIDATION ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= 50:  # 50 batches = ~3M tokens validation
            break
        x, y = x.to(device), y.to(device)
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

train_iter = iter(train_loader)

while step < CFG.total_steps:
    # ── Gradient accumulation loop ──────────────────────────────────────────
    accum_loss = 0.0

    for micro_step in range(CFG.grad_accum):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)  # reset dataloader (new epoch)
            x, y = next(train_iter)
            log.info(f"Step {step}: Starting new epoch over data")

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

        # Warning thresholds
        if avg_grad > 10.0:
            log.warning(f"⚠ High gradient norm: {avg_grad:.2f} — consider reducing LR")
        if vram_used > 180:
            log.warning(f"⚠ High VRAM usage: {vram_used:.0f}GB — OOM risk")
        if avg_loss > 8.0 and step > 1000:
            log.warning(f"⚠ Loss not decreasing at step {step} — check config")

        running_loss = 0.0
        running_grad_norm = 0.0

    # ── Validation ───────────────────────────────────────────────────────────
    if step % CFG.val_every == 0:
        val_loss, val_ppl = evaluate()
        log.info(f"  ▶ VAL step={step}: loss={val_loss:.4f}, ppl={val_ppl:.1f}")

        # Early warning: val loss diverging from train
        if step > 2000:
            recent_train = accum_loss  # approximate
            if val_loss > recent_train * 1.3:
                log.warning("⚠ Val loss >> train loss — possible overfitting or data issue")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if step % CFG.save_every == 0:
        val_loss, _ = evaluate()
        save_checkpoint(step, accum_loss, val_loss)

# ─── TRAINING COMPLETE ────────────────────────────────────────────────────────

total_time_h = (time.time() - t0) / 3600
log.info("\n" + "=" * 60)
log.info("TRAINING COMPLETE")
log.info(f"Total time: {total_time_h:.1f} hours")
log.info(f"Total tokens: {tokens_trained/1e9:.2f}B")
log.info(f"Final avg tok/sec: {tokens_trained / (total_time_h * 3600):,.0f}")
log.info("=" * 60)

# Final checkpoint
val_loss, val_ppl = evaluate()
save_checkpoint(step, accum_loss, val_loss)
log.info(f"Final val loss: {val_loss:.4f}, ppl: {val_ppl:.1f}")

# Save model in HuggingFace format for easy loading
hf_save_path = "/data/checkpoints/karnalm-pretrain/hf_model"
model.save_pretrained(hf_save_path)
log.info(f"Model saved in HF format: {hf_save_path}")

# Push final model and tokenizer to Hugging Face
if CFG.hf_repo_id:
    log.info("Pushing final model to Hugging Face hub...")
    try:
        model.push_to_hub(CFG.hf_repo_id)
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=CFG.tok_path)
        tok.push_to_hub(CFG.hf_repo_id)
        log.info(f"✓ Final model and tokenizer pushed to {CFG.hf_repo_id}")
    except Exception as e:
        log.error(f"⚠ Failed to push final model to HF hub: {e}")

log.info("Next: python3 training_scripts/04_sft.py")
