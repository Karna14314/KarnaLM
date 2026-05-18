# KarnaLM — Architectural Decisions & Pivot Log

This document serves as the historical record of all major roadblocks, struggles, and strategic pivots executed during the KarnaLM development session. It tracks why specific architectural decisions were made for the MI300X instance.

---

### 1. The ROCm 7.2.0 Matrix Math Failure
*   **The Struggle:** The initial environment was configured for PyTorch on ROCm 6.1. However, the host MI300X instance was running ROCm 7.2.0. This mismatch caused fatal `invalid device function` errors and `Segmentation fault` crashes during basic matrix multiplication tests. The system also blocked global pip installs via PEP 668.
*   **The Pivot:** We bypassed the system block by creating an isolated `venv` and forced an upgrade to `torch-2.5.1+rocm6.2`. This resolved the C++ kernel mismatch and permanently stabilized matrix operations on the MI300X.

### 2. The Flash Attention Build Wall
*   **The Struggle:** Building native `flash-attn` from C++ source failed completely—a notoriously common issue on AMD GPUs. 
*   **The Pivot:** We initially attempted to use Triton's Flash Attention backend. Ultimately, we pivoted entirely to PyTorch's native Scaled Dot Product Attention (`sdpa`). This proved to be mathematically equivalent, inherently stable on AMD, and vastly easier to deploy without compilation penalties.

### 3. Tokenizer Dilution & Multiprocessing
*   **The Struggle:** The raw parquet datasets were massive (117GB raw text) and single-threaded tokenization was too slow. More importantly, simply sampling data would dilute the vital Telugu corpus (only 5GB) against the massive English corpus (60GB).
*   **The Pivot:** We implemented a 64-core `ProcessPoolExecutor` script. To protect Telugu, we instituted hard caps (English: 10B, Hindi: 7B, Telugu: 5B). Because Telugu's cap was higher than its total volume, 100% of the Telugu corpus was consumed, resulting in a perfectly balanced, 18.8 Billion token trilingual dataset distributed across 19 binary shards.

### 4. The Budget Reality Check (1.1B vs 360M Pivot)
*   **The Struggle:** The original ambition was a 1.1B parameter LLaMA architecture. However, real-world smoke tests on the MI300X revealed a hard throughput ceiling of ~22,000 tokens/second for a 1.1B model. Math revealed that training 17.8B tokens at that speed would take **244 hours**, far exceeding the hard **95-hour ($190)** credit budget. We would run out of money at 18% completion, yielding a useless, undertrained model.
*   **The Pivot:** We executed a massive architectural pivot to a **Chinchilla-optimal 360M parameter model** (24 Layers, 1024d, 16H). This skyrocketed computational throughput to **~51,500 tok/s**. We resized the training target to 12.8B tokens, which perfectly saturates a 360M model in exactly **70 hours**, leaving 25 hours of safe buffer for Supervised Fine-Tuning (SFT). A fully converged 360M mathematically outperforms a 15%-trained 1.1B model in every metric.

### 5. Memory Exhaustion & File Descriptor Leaks
*   **The Struggle:** The `ShardedBinaryDataset` loader was opening `np.memmap` files on every single micro-batch. At 51,500 tok/s, it was opening 16 files per second. Over 70 hours, this 2.3 million file open/close cycle posed a severe risk of exhausting Linux's "Too Many Open Files" limit or triggering a hidden Python garbage collection leak.
*   **The Pivot:** We completely rewrote the dataloader to open the massive 2GB binary files exactly *once* during initialization, holding them as persistent mapped memory. Additionally, static sequence lengths (2048) locked the MI300X VRAM allocation safely at 9GB, completely eliminating the risk of a late-stage OOM crash.

### 6. Pipeline Automation & Idling Risks
*   **The Struggle:** Running Phase 1 (Pretrain) and Phase 2 (SFT) manually meant the GPU might idle overnight while waiting for human input, burning expensive credits.
*   **The Pivot:** We engineered an unattended `05_train_and_deploy.sh` script to run inside a detached `tmux` session. It handles pre-training, Hugging Face checkpoint syncing, seamless handoff to the SFT script, final Chat model deployment, and benchmarking. This allows the human operator to completely disconnect.
