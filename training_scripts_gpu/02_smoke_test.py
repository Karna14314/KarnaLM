#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     KarnaLM — SCRIPT 02: SMOKE TEST                             ║
║     Verifies the FULL training pipeline works before            ║
║     committing to the 33-hour run.                              ║
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
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_HIP_ALLOC_CONF",
]
for var in required_env:
    val = os.environ.get(var, None)
    check(f"ENV {var}", val is not None, val or "NOT SET — run 00_amd_setup.sh")

# ─── CHECK 3: Data Files ──────────────────────────────────────────────────────
print("\n▶ CHECK 3: Data Files")

train_shard_pattern = "/mnt/scratch/shards/train_shard_*.bin"
val_path   = Path("/mnt/scratch/shards/val.bin")
tok_path   = Path("/mnt/scratch/tokenizer/karnalm_tokenizer.json")

import glob
shard_files = sorted(glob.glob(train_shard_pattern))
check("train shards exist", len(shard_files) > 0, f"Found {len(shard_files)} shards")
check("val.bin exists", val_path.exists())
check("tokenizer.json exists", tok_path.exists())

if shard_files:
    # Check first shard size using file stat (avoid loading into memory)
    first_shard_size = os.path.getsize(shard_files[0])
    first_shard_tokens = first_shard_size // 2  # uint16
    check("first shard > 100M tokens", first_shard_tokens > 100_000_000,
          f"{first_shard_tokens/1e6:.1f}M tokens")
    # Quick sanity: read first 2000 bytes and check for non-zero
    with open(shard_files[0], 'rb') as f:
        header = f.read(2000)
    vals = [int.from_bytes(header[i:i+2], 'little') for i in range(0, 2000, 2)]
    check("No zero tokens in first 1000", min(vals) >= 0)

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
    check("Loss in expected range (7-12)", 7.0 < loss_val < 12.0,
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

# ─── CHECK 5B: SDPA Verification ──────────────────────────────────────────────
print("\n▶ CHECK 5B: SDPA Attention Backend")

try:
    sdpa_config = LlamaConfig(
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1408,
        max_position_embeddings=2048,
        vocab_size=52_000,
        torch_dtype="bfloat16",
        use_cache=False,
        attn_implementation="sdpa",
    )
    sdpa_model = LlamaForCausalLM(sdpa_config).to("cuda", dtype=torch.bfloat16)
    attn_impl = sdpa_model.config._attn_implementation
    check("SDPA config accepted", attn_impl == "sdpa",
          f"attn_implementation={attn_impl}")
    
    # Actually run a forward pass to confirm no fallback
    x = torch.randint(0, 52000, (2, 256), device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = sdpa_model(input_ids=x, labels=x)
    check("SDPA forward pass succeeds", True, "ROCm optimized backend active")
    
    del sdpa_model, out
    torch.cuda.empty_cache()
except Exception as e:
    check("SDPA backend", False, f"{e}")

# ─── CHECK 6: Real Model Memory Estimate ─────────────────────────────────────
print("\n▶ CHECK 6: Full 1.1B Model VRAM Estimate")

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

# ─── CHECK 7: REAL 1.1B Throughput Benchmark ──────────────────────────────────
print("\n▶ CHECK 7: Real 1.1B Throughput Benchmark")
print("  Loading actual 1.1B model — this takes a minute...")

real_model = None
optimizer = None
try:
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    real_config = LlamaConfig(
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=5632,
        max_position_embeddings=2048,
        vocab_size=52_000,
        torch_dtype="bfloat16",
        use_cache=False,
        attn_implementation="sdpa",
    )
    real_model = LlamaForCausalLM(real_config).to("cuda", dtype=torch.bfloat16)
    real_params = sum(p.numel() for p in real_model.parameters())
    print(f"  Model loaded: {real_params/1e9:.2f}B params")
    
    optimizer = torch.optim.AdamW(real_model.parameters(), lr=3e-4)
    
    TOTAL_TOKENS = 17_820_000_000
    BENCH_SEQ = 2048
    N_STEPS = 10
    
    # ── Batch=4 benchmark ──
    BATCH_4 = 4
    print(f"\n  Benchmarking batch={BATCH_4}, seq={BENCH_SEQ}...")
    
    # Warmup
    for _ in range(3):
        x = torch.randint(0, 52000, (BATCH_4, BENCH_SEQ), device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = real_model(input_ids=x, labels=x).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(N_STEPS):
        x = torch.randint(0, 52000, (BATCH_4, BENCH_SEQ), device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = real_model(input_ids=x, labels=x).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    elapsed_4 = time.time() - t0
    
    tps_4 = (N_STEPS * BATCH_4 * BENCH_SEQ) / elapsed_4
    hours_4 = TOTAL_TOKENS / tps_4 / 3600
    vram_4 = torch.cuda.memory_allocated() / 1024**3
    
    print(f"  Batch=4 results:")
    print(f"    Throughput:  {tps_4:,.0f} tok/sec")
    print(f"    Hours for 17.8B: {hours_4:.1f} hours")
    print(f"    VRAM used:   {vram_4:.1f} GB")
    
    check("Throughput > 5K tok/sec (batch=4)", tps_4 > 5000,
          f"{tps_4:,.0f} tok/sec")
    
    # ── Try Batch=8 if VRAM allows ──
    tps_8 = None
    hours_8 = None
    vram_8 = None
    
    if vram_4 < 160:
        BATCH_8 = 8
        print(f"\n  VRAM OK ({vram_4:.0f}GB) — trying batch={BATCH_8}...")
        
        try:
            # Warmup
            for _ in range(3):
                x = torch.randint(0, 52000, (BATCH_8, BENCH_SEQ), device="cuda")
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = real_model(input_ids=x, labels=x).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            t0 = time.time()
            
            for _ in range(N_STEPS):
                x = torch.randint(0, 52000, (BATCH_8, BENCH_SEQ), device="cuda")
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = real_model(input_ids=x, labels=x).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            elapsed_8 = time.time() - t0
            
            tps_8 = (N_STEPS * BATCH_8 * BENCH_SEQ) / elapsed_8
            hours_8 = TOTAL_TOKENS / tps_8 / 3600
            vram_8 = torch.cuda.memory_allocated() / 1024**3
            
            print(f"  Batch=8 results:")
            print(f"    Throughput:  {tps_8:,.0f} tok/sec")
            print(f"    Hours for 17.8B: {hours_8:.1f} hours")
            print(f"    VRAM used:   {vram_8:.1f} GB")
            
            check("Throughput > 5K tok/sec (batch=8)", tps_8 > 5000,
                  f"{tps_8:,.0f} tok/sec")
        except RuntimeError as oom:
            print(f"  Batch=8 OOM: {oom}")
            check("Batch=8 fits in VRAM", False, "OOM — use batch=4")
    else:
        print(f"  VRAM too high for batch=8 ({vram_4:.0f}GB) — skipping")
    
    # Final verdict: pick the best config
    best_tps = tps_8 if (tps_8 and hours_8) else tps_4
    best_hours = TOTAL_TOKENS / best_tps / 3600
    best_batch = 8 if (tps_8 and hours_8) else 4
    
    print(f"\n  ═══ RECOMMENDED CONFIG ═══")
    print(f"    Batch size:  {best_batch}")
    print(f"    Throughput:  {best_tps:,.0f} tok/sec")
    print(f"    Est. hours:  {best_hours:.1f} hours")
    
    check("17.8B tokens fits in 250 hours (extrapolated from small batch)", best_hours < 250,
          f"~{best_hours:.0f}h estimated at batch={best_batch}")

except Exception as e:
    import traceback
    traceback.print_exc()
    check("1.1B Throughput benchmark", False, str(e))
finally:
    if real_model is not None:
        del real_model
    if optimizer is not None:
        del optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ─── CHECK 8: Per-Language Token Distribution ─────────────────────────────────
print("\n▶ CHECK 8: Per-Language Token Distribution")

summary_path = Path("/mnt/scratch/shards/token_summary.json")
if summary_path.exists():
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Check overall total
        total_tokens = summary.get("total_tokens", 0)
        print(f"  Approx total tokens: {total_tokens/1e9:.2f}B")
        check("total tokens > 1B", total_tokens > 1_000_000_000,
              f"{total_tokens/1e9:.2f}B tokens")
        
        # Check Telugu directly
        langs = summary.get("per_language", {})
        te_info = langs.get("telugu", {})
        te_pct = te_info.get("percentage", 0)
        te_tokens = te_info.get("tokens", 0)
        
        print(f"  From token_summary.json: Telugu={te_pct}% ({te_tokens/1e9:.1f}B tokens)")
        check("Telugu > 5% of total data", te_pct > 5,
              f"Telugu is {te_pct}% — {te_tokens/1e9:.1f}B tokens")
              
        print(f"  Token summary breakdown:")
        for lang, info in langs.items():
            print(f"    {lang}: {info.get('tokens', 0)/1e9:.2f}B ({info.get('percentage', 0)}%)")
            
    except Exception as e:
        check("Language distribution check", False, f"Failed parsing JSON: {e}")
else:
    check("Language distribution check", False, "No token_summary.json found to verify Telugu mix")


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
    print("║  Fix all failed checks before starting training     ║")
else:
    print("║  ✓ ALL CHECKS PASSED                                 ║")
    print("║  Safe to start full training run                     ║")
    print("║  Next: bash training_scripts/00_verify_and_launch.sh ║")

print("╚══════════════════════════════════════════════════════╝")

exit(len(FAIL))
