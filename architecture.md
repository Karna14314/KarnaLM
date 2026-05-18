# KarnaLM — Comprehensive Architecture & Strategy

> **Revision:** 3.0 (Post-1.1B OOM Pivot) | **Hardware:** AMD Instinct MI300X (192GB HBM3)

---

## 📊 1. Data Ecosystem

### Data Sources & Volumes
| Source | Language | Target Tokens |
| :--- | :--- | :--- |
| **FineWeb** | English | ~6.5B |
| **CC-100 (hi)** | Hindi | ~3.5B |
| **CC-100 (te)** | Telugu | ~1.0B |
| **Aya + FLAN** | Mixed | SFT Phase |
| **Total** | | **~12.8B Target / 17.8B Available** |

### Data Processing Strategy
*   **Parallel Tokenization:** Uses all available CPU cores via `ProcessPoolExecutor`.
*   **Sharded Binary Format:** `train_shard_XXXX.bin` files dynamically loaded via `np.memmap`.
*   **Random Interleaved Loading:** `ShardedBinaryDataset` randomly shuffles shards to prevent catastrophic forgetting of specific languages.
*   **Memory Safe Mapping:** Memmaps are loaded once in `__init__` to prevent file descriptor leaks over 70 hours.

---

## 🏗️ 2. Model Architecture (KarnaLM-360M)

Pivot from 1.1B to Chinchilla-optimal 360M for deep, thorough training on 12.8B tokens within the 70-hour compute budget.

| Parameter | Specification | Rationale |
| :--- | :--- | :--- |
| **Total Params** | ~390M | 337M core + 53M embeddings (52K vocab). |
| **Hidden Dim** | 1024 | Scaled down for 360M parameter target. |
| **Layers** | 24 | Optimized for depth/reasoning. |
| **Attention** | GQA (16Q / 8KV) | PyTorch native `sdpa` for stable ROCm throughput. |
| **Activation** | SwiGLU (2816) | Standard LLaMA-style FFN. |
| **Vocab Size** | 52,000 | Comprehensive EN+HI+TE BPE coverage. |
| **PE** | RoPE (θ=10000) | Standard rotary embeddings. |
| **Memory Opt** | **Disabled** | Gradient Checkpointing disabled for maximum tok/s. |

---

## 🚀 3. Training Infrastructure (AMD Optimized)

### Hardware & Software
*   **Platform:** ROCm 6.1 (MI300X).
*   **Precision:** Pure **BFloat16** (Native MI300X support).
*   **Attention Backend:** **PyTorch SDPA** (Flash Attention 2 proved unstable/unnecessary).
*   **Memory Management:** `expandable_segments:True` and `garbage_collection_threshold:0.8` (though expandable segments is bypassed on current ROCm). VRAM sits safely at 9GB.

### The Automated Pipeline (`05_train_and_deploy.sh`)
An unattended, resumable script that chains the entire lifecycle:
1.  **Step 1:** Pre-train 12.8B tokens (`03_pretrain.py`).
2.  **Step 2:** Push base model to Hugging Face (`ncncomplete/KarnaLM-360M-base`).
3.  **Step 3:** Supervised Fine-Tuning (`04_sft.py`).
4.  **Step 4:** Push chat model to Hugging Face (`ncncomplete/KarnaLM-360M-chat`).
5.  **Step 5:** Inference verification benchmark.

---

## 💰 4. Budget & Training Specs ($190 Credit)

| Metric | Specification |
| :--- | :--- |
| **Time Budget** | ~95 hours total (~70h Pretrain + ~15h SFT) |
| **Target Tokens**| 12.8 Billion tokens (72% of dataset) |
| **Throughput** | ~51,500 tokens / sec |
| **Peak LR** | 5e-4 (Cosine decay to 5e-5) |
| **Global Batch** | 16 batch × 4 grad_accum = 64 effective (~131K tokens/step) |
| **Total Steps** | 97,656 steps |
| **Checkpoints** | Save local every 2000 |
| **HF Sync** | Push to Hub every 10,000 steps |

---

## ✅ 5. Operational Health Checks (Verified)

- [x] **OOM Proof:** Static batching locks VRAM at 9GB / 192GB. No gradient checkpointing needed.
- [x] **File Descriptor Leak Proof:** Memmaps loaded persistently in data loader.
- [x] **Storage Proof:** `keep_ckpts=3` logic automatically deletes old `.pt` files.
- [x] **Resume Proof:** `03_pretrain.py` automatically detects and resumes from latest checkpoint.
- [x] **SFT Path Proof:** `04_sft.py` correctly points to the `360m/final` base model.
