# KarnaLM — Comprehensive Architecture & Strategy

> **Revision:** 2.0 (Post-Audit) | **Hardware:** AMD Instinct MI300X (192GB HBM3)

---

## 📊 1. Data Ecosystem

### Data Sources & Volumes
| Source | Language | Parquet Size | Est. Raw Text | Target Tokens |
| :--- | :--- | :--- | :--- | :--- |
| **FineWeb** | English | 16.6 GB | ~60 GB | ~6.5B |
| **CC-100 (hi)** | Hindi | 12.0 GB | ~40 GB | ~3.5B |
| **CC-100 (te)** | Telugu | 5.0 GB | ~12 GB | ~1.0B |
| **Aya + FLAN** | Mixed | 2.0 GB | ~5 GB | SFT Phase |
| **Total** | | **35.6 GB** | **~117 GB** | **~11.0B** |

### Data Processing Strategy
*   **Parallel Tokenization:** Uses all available CPU cores (64-128 on MI300X host) via `ProcessPoolExecutor`.
*   **Sharded Binary Format:** Outputs `train_shard_XXXX.bin` files. Each shard is exactly **1B tokens (~2GB)**.
*   **Disk Safety:** Implements a global **11B token cap** (22GB binary) to prevent disk overflow and maintain a 12B token training horizon.
*   **Hugging Face Backup:** Integrated `huggingface_hub` for asynchronous, non-blocking checkpoint uploads to `KarnaDigital/KarnaLM-1.1B-chat`.
*   **Deduplication:** MinHash deduplication is recommended for CC-100 sources to improve generalization.

---

## 🏗️ 2. Model Architecture (KarnaLM-1.1B)

| Parameter | Specification | Rationale |
| :--- | :--- | :--- |
| **Hidden Dim** | 2048 | Balanced for 1.1B total params. |
| **Layers** | 22 | Optimized for depth/reasoning. |
| **Attention** | GQA (32Q / 8KV) | 4x KV cache reduction for throughput. |
| **Activation** | SwiGLU (5632) | Industry standard for LLaMA-style models. |
| **Vocab Size** | 52,000 | Comprehensive EN+HI+TE BPE coverage. |
| **PE** | RoPE (θ=10000) | Standard rotary embeddings. |
| **Norm** | RMSNorm (ε=1e-5) | Stability in BF16 training. |
| **Memory Opt** | **Gradient Checkpointing** | Enabled to allow larger batch sizes without OOM. |

---

## 🚀 3. Training Infrastructure (AMD Optimized)

### Hardware & Software
*   **Platform:** ROCm 6.1 (Latest stable for MI300X).
*   **Precision:** Pure **BFloat16** (Native MI300X support).
*   **Attention Backend:** **Flash Attention 2** via Triton (Verified).
*   **Kernel Tuning:** **TunableOp** enabled (Cache saved to `/data/tunableop_cache.csv`).
*   **Memory Management:** `PYTORCH_HIP_ALLOC_CONF` set to `max_split_size_mb:128` to prevent fragmentation.

### The Pipeline
1.  **Stage 0: Setup (`00_amd_setup.sh`)**: Env vars (including `HF_TOKEN`) + ROCm libs + TunableOp.
2.  **Stage 1: Parallel Tokenization (`01_tokenize_data.py`)**: 64-core processing + sharding (1B tokens per shard).
3.  **Stage 2: Smoke Test (`02_smoke_test.py`)**: FA2 verification + throughput benchmark + VRAM check.
4.  **Stage 3: Pre-training (`03_pretrain.py`)**: 32-batch run on 11B tokens with **Asynchronous Cloud Backups**.
5.  **Stage 4: SFT (`04_sft.py`)**: Chat-alignment using Aya and FLAN.

---

## 💰 4. Budget & Training Specs ($190 Credit)

| Metric | Specification |
| :--- | :--- |
| **Peak LR** | 3e-4 (Cosine decay to 3e-5) |
| **Warmup** | 2000 steps |
| **Global Batch Size** | 128 sequences (32 batch × 4 grad_accum) |
| **Effective Tokens/Step** | ~262,144 tokens |
| **Checkpoint Freq** | Every 1000 steps (Auto-pushed to HF) |
| **Estimated Runtime** | ~32-35 hours (Pre-training) |

---

## ✅ 5. Operational Checkpoints (Pre-Flight)

- [ ] **Core Count:** `01_tokenize_data.py` must use `os.cpu_count()`.
- [ ] **Shard Verification:** `train_shard_*.bin` files must be exactly 2GB each.
- [ ] **FA2 Check:** `02_smoke_test.py` must explicitly confirm `flash_attention_2` implementation.
- [ ] **OOM Prevention:** `gradient_checkpointing` must be `True` in `03_pretrain.py`.
- [ ] **HF Login:** Ensure `HF_TOKEN` is exported in the setup script.
