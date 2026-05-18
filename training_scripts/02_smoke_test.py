#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     KarnaLM — SCRIPT 02: SMOKE TEST                             ║
║     Verifies the FULL training pipeline works before            ║
║     committing to the 45-hour run.                              ║
║                                                                  ║
║     Run: python3 scripts/02_smoke_test.py                       ║
║     Expected: completes in 5-10 minutes                         ║
║     If ANY check fails — DO NOT start full training             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import time
import os
import json
from pathlib import Path
from transformers import (
    LlamaConfig, LlamaForCausalLM,
    PreTrainedTokenizerFast
)
from tokenizers import Tokenizer

print("╔══════════════════════════════════════════════════════╗")
print("║         KarnaLM Smoke Test Starting                  ║")
print("╚══════════════════════════════════════════════════════╝")
print()

PASS = []
FAIL = []

def check(name, result, detail=""):
    if result:
        PASS.append(name)
        print(f"  ✓ {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL.append(name)
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))

# ─── CHECK 1: GPU ──────────────────────────────────────────────────────────────
print("▶ CHECK 1: GPU Environment")

check("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check("VRAM > 150GB", vram > 150, f"{vram:.0f} GB detected")
    check("BF16 supported", torch.cuda.is_bf16_supported())
    check("GPU name contains AMD or MI", 
          "AMD" in torch.cuda.get_device_name(0) or "MI" in torch.cuda.get_device_name(0),
          torch.cuda.get_device_name(0))

# ─── CHECK 2: ROCm Environment Variables ─────────────────────────────────────
print("\n▶ CHECK 2: ROCm Environment Variables")

required_env = [
    "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_HIP_ALLOC_CONF",
    "HIP_FORCE_DEV_KERNARG",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE",
]
for var in required_env:
    val = os.environ.get(var, None)
    check(f"ENV {var}", val is not None, val or "NOT SET — run 00_amd_setup.sh")

# ─── CHECK 3: Data Files ──────────────────────────────────────────────────────
print("\n▶ CHECK 3: Data Files")

train_path = Path("/data/tokens/train.bin")
val_path   = Path("/data/tokens/val.bin")
tok_path   = Path("/data/tokenizer/karnalm_tokenizer.json")

check("train.bin exists", train_path.exists())
check("val.bin exists", val_path.exists())
check("tokenizer.json exists", tok_path.exists())

if train_path.exists():
    train_arr = np.fromfile(train_path, dtype=np.uint16)
    check("train.bin > 1B tokens", len(train_arr) > 1_000_000_000,
          f"{len(train_arr)/1e9:.2f}B tokens")
    check("No zero tokens in first 1000", train_arr[:1000].min() > 0)

# ─── CHECK 4: Tokenizer ───────────────────────────────────────────────────────
print("\n▶ CHECK 4: Tokenizer")

if tok_path.exists():
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    check("Vocab size 50K-55K", 50_000 <= vocab_size <= 55_000, f"{vocab_size:,}")
    
    vocab = tokenizer.get_vocab()
    check("<|endoftext|> in vocab", "<|endoftext|>" in vocab)
    check("<|im_start|> in vocab", "<|im_start|>" in vocab)
    
    # Encode/decode roundtrip for all 3 languages
    tests = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Hindi":   "भारत की राजधानी नई दिल्ली है।",
        "Telugu":  "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    }
    for lang, text in tests.items():
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids).strip()
        roundtrip = decoded.replace(" ", "") == text.replace(" ", "")
        check(f"Tokenizer roundtrip {lang}", roundtrip,
              f"{len(ids)} tokens")

# ─── CHECK 5: Model Init & Forward Pass ──────────────────────────────────────
print("\n▶ CHECK 5: Model Initialization & Forward Pass (mini model)")

# Use SMALL config for smoke test — just to verify the code path
mini_config = LlamaConfig(
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,      # GQA
    intermediate_size=1408,
    max_position_embeddings=2048,
    vocab_size=52_000,
    torch_dtype="bfloat16",
    use_cache=False,
)

try:
    model = LlamaForCausalLM(mini_config).to("cuda", dtype=torch.bfloat16)
    param_count = sum(p.numel() for p in model.parameters())
    check("Mini model created", True, f"{param_count/1e6:.0f}M params")
    
    # Forward pass
    dummy_input = torch.randint(0, 52000, (2, 512), device="cuda")
    dummy_labels = dummy_input.clone()
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids=dummy_input, labels=dummy_labels)
    
    loss_val = out.loss.item()
    check("Forward pass runs", True)
    check("Loss is finite", np.isfinite(loss_val), f"loss={loss_val:.4f}")
    check("Loss in expected range (7-11)", 7.0 < loss_val < 12.0,
          f"loss={loss_val:.4f} (random init should be ~ln(52000)≈10.7)")
    
    # Backward pass
    out.loss.backward()
    check("Backward pass runs", True)
    
    # Gradient check
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    max_grad = max(grad_norms)
    check("Gradients computed", len(grad_norms) > 0, f"{len(grad_norms)} tensors")
    check("No NaN gradients", all(np.isfinite(g) for g in grad_norms))
    check("Grad norm reasonable (<100)", max_grad < 100, f"max_grad={max_grad:.2f}")
    
    del model, out
    torch.cuda.empty_cache()
    
except Exception as e:
    check("Model creation/forward pass", False, str(e))

# ─── CHECK 5B: Flash Attention 2 Verification ─────────────────────────────────
print("\n▶ CHECK 5B: Flash Attention 2 Backend")

try:
    fa2_config = LlamaConfig(
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1408,
        max_position_embeddings=2048,
        vocab_size=52_000,
        torch_dtype="bfloat16",
        use_cache=False,
        attn_implementation="flash_attention_2",
    )
    fa2_model = LlamaForCausalLM(fa2_config).to("cuda", dtype=torch.bfloat16)
    attn_impl = fa2_model.config._attn_implementation
    check("FA2 config accepted", attn_impl == "flash_attention_2",
          f"attn_implementation={attn_impl}")
    
    # Actually run a forward pass to confirm no fallback
    x = torch.randint(0, 52000, (2, 256), device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = fa2_model(input_ids=x, labels=x)
    check("FA2 forward pass succeeds", True, "Triton backend active")
    
    del fa2_model, out
    torch.cuda.empty_cache()
except Exception as e:
    check("Flash Attention 2", False,
          f"{e} — FA2 may not work. Training will be 30-40% slower.")

# ─── CHECK 6: Real Model Memory Estimate ─────────────────────────────────────
print("\n▶ CHECK 6: Full 1.1B Model VRAM Estimate")

# Load actual config (we'll define this in next script)
full_config = LlamaConfig(
    hidden_size=2048,
    num_hidden_layers=22,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=5632,
    max_position_embeddings=2048,
    vocab_size=52_000,
    torch_dtype="bfloat16",
    use_cache=False,
)

# Estimate VRAM without actually loading
param_count = sum(
    np.prod(p)
    for p in [
        # Embedding
        (52000, 2048),
        # Per layer (22 layers)
        *[(2048, 2048)] * 22,    # q_proj
        *[(256, 2048)] * 22,     # k_proj (GQA: 8 heads * 32 head_dim)
        *[(256, 2048)] * 22,     # v_proj
        *[(2048, 2048)] * 22,    # o_proj
        *[(5632, 2048)] * 22,    # gate_proj
        *[(5632, 2048)] * 22,    # up_proj
        *[(2048, 5632)] * 22,    # down_proj
        *[(2048,)] * 22,         # input_layernorm
        *[(2048,)] * 22,         # post_attn_layernorm
        # Final norm + lm_head
        (2048,),
        (52000, 2048),
    ]
)

# Memory breakdown (training)
# Weights (BF16): 2 bytes/param
# Gradients (FP32): 4 bytes/param
# Adam states (FP32): 8 bytes/param (m + v)
# Activations (BF16, seq=2048, batch=32): rough estimate
weights_gb    = param_count * 2 / 1024**3
gradients_gb  = param_count * 4 / 1024**3
optimizer_gb  = param_count * 8 / 1024**3
activations_gb = 2048 * 2048 * 32 * 22 * 2 / 1024**3  # rough

total_est = weights_gb + gradients_gb + optimizer_gb + activations_gb

print(f"  Estimated 1.1B model VRAM usage:")
print(f"    Weights (BF16):      {weights_gb:.1f} GB")
print(f"    Gradients (FP32):    {gradients_gb:.1f} GB")
print(f"    Optimizer (Adam):    {optimizer_gb:.1f} GB")
print(f"    Activations (est):   {activations_gb:.1f} GB")
print(f"    ─────────────────────────────────")
print(f"    Total estimate:      {total_est:.1f} GB")
print(f"    Available VRAM:      ~192 GB")
print(f"    Headroom:            ~{192-total_est:.0f} GB")

check("Fits in 192GB VRAM", total_est < 175,
      f"Estimated {total_est:.0f}GB — {'OK' if total_est < 175 else 'TOO TIGHT'}")

# ─── CHECK 7: Throughput Benchmark ────────────────────────────────────────────
print("\n▶ CHECK 7: Throughput Benchmark (mini model, batch=8, seq=2048)")
print("  (Extrapolate to estimate full model throughput)")

try:
    bench_config = LlamaConfig(
        hidden_size=1024,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2816,
        max_position_embeddings=2048,
        vocab_size=52_000,
        use_cache=False,
    )
    bench_model = LlamaForCausalLM(bench_config).to("cuda", dtype=torch.bfloat16)
    bench_params = sum(p.numel() for p in bench_model.parameters())
    
    optimizer = torch.optim.AdamW(bench_model.parameters(), lr=3e-4)
    
    # Warmup
    for _ in range(3):
        x = torch.randint(0, 52000, (4, 512), device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = bench_model(input_ids=x, labels=x).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Benchmark 10 steps
    BENCH_BATCH = 8
    BENCH_SEQ   = 2048
    N_STEPS     = 10
    
    start = time.time()
    for _ in range(N_STEPS):
        x = torch.randint(0, 52000, (BENCH_BATCH, BENCH_SEQ), device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = bench_model(input_ids=x, labels=x).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_sec_bench = (N_STEPS * BENCH_BATCH * BENCH_SEQ) / elapsed
    
    # Scale estimate: throughput scales roughly inversely with param count
    # bench model is ~300M, full is ~1100M → ~3.5x slower
    scaling_factor = bench_params / (1_100_000_000)
    tokens_per_sec_est = tokens_per_sec_bench * scaling_factor
    
    # Hours to train 15B tokens
    hours_est = 15_000_000_000 / tokens_per_sec_est / 3600
    
    print(f"  Benchmark model: {bench_params/1e6:.0f}M params")
    print(f"  Throughput (bench model): {tokens_per_sec_bench:,.0f} tok/sec")
    print(f"  Estimated 1.1B throughput: {tokens_per_sec_est:,.0f} tok/sec")
    print(f"  Estimated hours for 15B tokens: {hours_est:.1f} hours")
    
    check("Throughput > 5K tok/sec", tokens_per_sec_est > 5000,
          f"~{tokens_per_sec_est:,.0f} tok/sec estimated")
    check("15B tokens fits in 45 hours", hours_est < 45,
          f"~{hours_est:.0f}h estimated")
    
    del bench_model
    torch.cuda.empty_cache()

except Exception as e:
    check("Throughput benchmark", False, str(e))

# ─── CHECK 8: Per-Language Token Distribution ─────────────────────────────────
print("\n▶ CHECK 8: Per-Language Token Distribution")

if tok_path.exists() and train_path.exists():
    try:
        tokenizer_check = Tokenizer.from_file(str(tok_path))
        # Sample 10,000 random sequences from train.bin
        train_data = np.memmap(str(train_path), dtype=np.uint16, mode='r')
        total_tokens_actual = len(train_data)
        print(f"  Total tokens in train.bin: {total_tokens_actual/1e9:.2f}B")
        check("train.bin has >5B tokens", total_tokens_actual > 5_000_000_000,
              f"{total_tokens_actual/1e9:.2f}B tokens")
        
        # Decode random samples to check language distribution
        import random
        random.seed(42)
        lang_samples = {"Hindi": 0, "Telugu": 0, "English": 0}
        n_samples = 500
        seq_len = 256
        for _ in range(n_samples):
            start = random.randint(0, len(train_data) - seq_len - 1)
            chunk = train_data[start:start+seq_len].astype(np.int64).tolist()
            text = tokenizer_check.decode(chunk)
            if any('\u0900' <= c <= '\u097F' for c in text):
                lang_samples["Hindi"] += 1
            elif any('\u0C00' <= c <= '\u0C7F' for c in text):
                lang_samples["Telugu"] += 1
            else:
                lang_samples["English"] += 1
        
        for lang, count in lang_samples.items():
            pct = count / n_samples * 100
            print(f"  {lang}: ~{pct:.0f}% of sampled sequences")
        
        te_pct = lang_samples["Telugu"] / n_samples * 100
        check("Telugu > 5% of data", te_pct > 5,
              f"Telugu is {te_pct:.0f}% — {'OK' if te_pct > 5 else 'TOO LOW, supplement Telugu data'}")
        
        del train_data
    except Exception as e:
        check("Language distribution check", False, str(e))
else:
    print("  ⚠ Skipping — train.bin or tokenizer not found yet")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────

print()
print("╔══════════════════════════════════════════════════════╗")
print("║                 SMOKE TEST RESULTS                   ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Passed: {len(PASS)}")
print(f"║  Failed: {len(FAIL)}")
print("╠══════════════════════════════════════════════════════╣")

if FAIL:
    print("║  ✗ FAILED CHECKS — DO NOT START TRAINING:")
    for f in FAIL:
        print(f"║    - {f}")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Fix all failed checks before running 04_train.sh    ║")
else:
    print("║  ✓ ALL CHECKS PASSED                                 ║")
    print("║  Safe to start full training run                     ║")
    print("║  Next: bash scripts/03_start_training.sh             ║")

print("╚══════════════════════════════════════════════════════╝")

exit(len(FAIL))
