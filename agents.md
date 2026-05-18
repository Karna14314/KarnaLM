# KarnaLM — Agent Execution Guide

This document defines the autonomous mission plan for the AI agent responsible for training **KarnaLM-360M-Chat**.

## 🎯 Primary Objective
Deliver a trilingual (EN, HI, TE) chat-aligned assistant model within the **$190 / 95-hour** credit window on an **AMD MI300X** instance.

---

## 🛤️ Phase 1: Environment & Data Ingestion (T+0 to T+3h)

### Task 1: Environment Setup
1.  Execute `bash training_scripts/00_amd_setup.sh`.
2.  **Verification:** Check `setup_log.txt`.
3.  Ensure all ROCm env vars are active. `PYTORCH_HIP_ALLOC_CONF` set to `expandable_segments:True,garbage_collection_threshold:0.8`.

### Task 2: Parallel Tokenization
1.  Execute `python3 training_scripts/01_tokenize_data.py`.
2.  **Monitor:** Confirm CPU usage spikes to 100% across all cores.
3.  **Verification:** Check `/mnt/scratch/shards/` for `train_shard_*.bin` files. Target is ~12.8B tokens (72% of full data).

### Task 3: Smoke Test (The Gatekeeper)
1.  Execute `python3 training_scripts/02_smoke_test.py`.
2.  **CRITICAL:** Flash Attention 2 is disabled due to platform instability. Native `sdpa` must be verified.
3.  Verify estimated training time fits the 70h budget (~51k tok/s).

---

## 🛤️ Phase 2: Fully Automated Pipeline (T+3h to T+85h)

### Task 4: Launch Training & Deployment
1.  Run `tmux new -s karnalm`.
2.  Execute `bash training_scripts/05_train_and_deploy.sh`.
3.  **This single script automates:**
    - `03_pretrain.py`: Pretrains the 360M model for 12.8B tokens (~70h).
    - HF Base Push: Uploads to `ncncomplete/KarnaLM-360M-base`.
    - `04_sft.py`: Fine-tunes using Aya and FLAN datasets (~12h).
    - HF Chat Push: Uploads to `ncncomplete/KarnaLM-360M-chat`.
    - Benchmark: Runs basic local inference verification.
4.  **Detaching:** Press `Ctrl+B`, then `D`.

### Task 5: Stability Monitoring
1.  Periodically check `tail -f /data/logs/pipeline.log`.
2.  Check live metrics at `tail -f /data/logs/training.log`.
    - `vram` should be perfectly stable at 9GB.
    - `tok/s` should be ~51,500.

---

## 🛠️ Troubleshooting for Agents

| Issue | Action |
| :--- | :--- |
| **Out of Memory (OOM)** | Verified impossible with 360M static batching. If it occurs, check for hidden list accumulations in Python. |
| **Loss Spikes / NaN** | The pipeline resumes automatically. It will load the last 2000-step checkpoint. |
| **Disk Full** | Verified impossible. `keep_ckpts=3` and HF overwrites same folder. Uses ~15GB max. |
| **Slow Throughput** | Ensure `attn_implementation="sdpa"`. Gradient checkpointing must be `False`. |

---

## 📝 Change Log & Versioning
- **Current Architecture:** 360M LLaMA-style (24 layers, 1024d, 16H).
- **Data Target:** 12.8B tokens.
- **Automation:** Fully unattended 5-stage pipeline in `05_train_and_deploy.sh`.
- **Memory Fixes:** Persistent memmap dataset loading implemented to stop file descriptor exhaustion.
