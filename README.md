# KarnaLM — 360M Parameter Trilingual Model

A Chinchilla-optimal 360M parameter LLaMA model trained on 12.8 Billion tokens of English, Hindi, and Telugu data. Optimized for AMD Instinct MI300X.

## 🚀 How to Check Progress

Use this single command to see exactly what the model is doing right now:

```bash
tail -n 20 /data/logs/training.log
```

Or for a live, auto-updating dashboard (press `Ctrl+C` to exit):

```bash
watch -n 30 bash training_scripts/monitor.sh
```

## 🏗️ Technical Overview
*   **Architecture:** 24 Layers, 1024 Hidden Dim, 16 Heads (390M Params total).
*   **Throughput:** ~51,000 tokens/second.
*   **Hardware:** AMD MI300X (VRAM usage: 9GB).
*   **Dataset:** 18.8B tokens processed, 12.8B tokens target training.

## 🛤️ Pipeline Status
The automated pipeline (`05_train_and_deploy.sh`) handles:
1.  **Pre-training:** 70 hours (~12.8B tokens).
2.  **Base Push:** Automatic upload to HuggingFace.
3.  **SFT:** Supervised Fine-tuning on Aya & FLAN datasets.
4.  **Chat Push:** Final model deployment.
5.  **Benchmark:** Local inference verification.

## 📚 Documentation
*   [Architectural Decisions](./architectural_decisions_log.md) — Why we pivoted from 1.1B to 360M.
*   [Engineering Learning Notes](./engineering_learning_notes.md) — Deep dives into VRAM, Speed, and Algorithms.
*   [Agent Guide](./agents.md) — Detailed execution steps for autonomous agents.
