import torch, time, gc, os
from transformers import LlamaConfig, LlamaForCausalLM

os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
# Remove expandable_segments — it's not supported and may cause allocator issues
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

# Aggressively clear any residual state from previous benchmarks
torch.cuda.empty_cache()
gc.collect()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"VRAM free at start: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
print()

config = LlamaConfig(
    hidden_size=1024, num_hidden_layers=24,
    num_attention_heads=16, num_key_value_heads=8,
    intermediate_size=2816, vocab_size=52000,
    max_position_embeddings=2048,
    attn_implementation="sdpa", use_cache=False,
)

print("360M model — batch sweep (no gradient checkpointing)")
print(f"{'Batch':>6} | {'Tok/s':>10} | {'VRAM':>7} | {'Full (17.8B)':>12} | {'Tokens in 45h':>14} | {'Tokens in 95h':>14}", flush=True)
print("-" * 82, flush=True)

for batch in [16, 32, 64, 128, 256, 512]:
    torch.cuda.empty_cache(); gc.collect()
    try:
        model = LlamaForCausalLM(config).to("cuda", dtype=torch.bfloat16)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        x = torch.randint(0, 52000, (batch, 2048), device="cuda")

        # warmup 3 steps
        for _ in range(3):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        tps = (10 * batch * 2048) / (time.time() - t0)
        vram = torch.cuda.max_memory_allocated() / 1024**3
        hours_full = 17_820_000_000 / tps / 3600
        tok_45h  = tps * 45 * 3600 / 1e9
        tok_95h  = tps * 95 * 3600 / 1e9

        print(f"{batch:>6} | {tps:>10,.0f} | {vram:>6.1f}GB | {hours_full:>10.1f}h | {tok_45h:>12.2f}B | {tok_95h:>12.2f}B", flush=True)

        del model, opt, x
        torch.cuda.empty_cache(); gc.collect()

    except torch.cuda.OutOfMemoryError as e:
        print(f"{batch:>6} | OOM — {e}", flush=True)
        torch.cuda.empty_cache(); gc.collect()
        break
    except Exception as e:
        print(f"{batch:>6} | Error: {e}", flush=True)
        break

print("\nDone.", flush=True)
