# KarnaLM — Master Forensic Audit Report & Engineering Blueprint

> **System Identity:** KarnaLM-360M-Chat
> **Architecture:** Chinchilla-Optimal Trilingual LLaMA (24 Layers, 1024 Hidden Dim, 16 Q-Heads, 8 KV-Heads / GQA)
> **Hardware Target:** AMD Instinct MI300X (192GB HBM3 VRAM, CDNA3 / gfx942)
> **Dataset Scale:** 18.82 Billion Tokens Processed / 12.8 Billion Tokens Pretraining Target
> **Execution Runtime:** ~82 Hours Fully Automated Pipeline ($190 Credit Budget)

---

# Phase 1 — Executive Project Reconstruction

### Problem Solved
Existing open-source Large Language Models (LLMs) suffer from acute **multilingual disparity**:
1. **Tokenizer Inefficiency (High Fertility):** Standard tokenizers (e.g., LLaMA-2, GPT-4) represent South Asian scripts (specifically Telugu and Hindi) inefficiently, requiring up to 15 tokens per word. This artificially inflates sequence lengths, destroys effective context window length, and increases compute costs by 3-4x.
2. **Resource Allocation Imbalance:** English heavily dominates public pretraining datasets. Naive dataset sampling results in severe language dilution for regional Indic languages.
3. **Hardware Monopoly Bottlenecks:** Most LLM pretraining pipelines are strictly locked to NVIDIA CUDA ecosystems. Demonstrating high-throughput, stable, cost-efficient LLM pretraining on AMD Instinct GPUs under ROCm requires novel pipeline engineering.

KarnaLM solves this by delivering a custom-tokenized, Chinchilla-optimal trilingual model (English, Hindi, Telugu) pre-trained and fine-tuned on AMD MI300X hardware within a strict **$190 / 95-hour compute budget**.

### Why It Was Created
KarnaLM was developed as a production-grade demonstration of:
- Building high-quality, parameter-efficient regional language models for undertrained languages (Telugu & Hindi).
- Maximizing compute budget efficiency by pivoting from an undertrained 1.1B parameter architecture to a fully converged, saturated **360M parameter architecture** running at **~51,500 tokens/second**.
- Establishing a stable, fault-tolerant training and deployment pipeline on AMD ROCm architecture.

### Target Use Case & End Users
- **Real-Time Trilingual Conversational AI:** Customer support, voice assistants, and educational interactive bots operating in English, Hindi, and Telugu.
- **Edge & Low-Latency Deployment:** A 360M parameter footprint (720MB FP16 / 360MB INT8) allows sub-10ms token generation latency on consumer hardware or mobile edge devices.
- **End Users:** Native speakers of Telugu and Hindi seeking natural conversational AI in their native scripts, enterprise developers needing small/fast Indic models, and ML researchers benchmarking ROCm hardware.

### Expected Outcomes & Core Innovation
- **Vocabulary Efficiency:** Telugu fertility reduced from ~15.0 tokens/word (standard LLaMA) down to **3.9 tokens/word**.
- **Model Convergence:** Complete training on **12.8 Billion tokens** (~35.5 tokens per parameter), meeting Chinchilla optimality for a 360M model.
- **Capped Multilingual Sampling Strategy:** Hard-capped dataset sampling pipeline guaranteeing 100% consumption of the 5GB Telugu corpus (1.5B tokens), combined with 35% Hindi and 50% English data.
- **AMD ROCm Kernel Optimization:** Complete elimination of native C++ compilation traps by utilizing PyTorch's native Scaled Dot Product Attention (SDPA) C++ backend, yielding native throughput of ~51,500 tok/s.
- **Zero-Leak Memory-Mapped Dataloader:** Custom binary dataset loader (`ShardedBinaryDataset`) holding persistent `np.memmap` handles, eliminating OS file descriptor exhaustion.

### Technical Stack
- **Hardware:** AMD Instinct MI300X (192GB HBM3 VRAM, 5.3 TB/s memory bandwidth, CDNA3 architecture / gfx942).
- **Driver & OS:** Ubuntu 22.04 LTS, AMD ROCm 7.2.0 driver host runtime with PyTorch `2.5.1+rocm6.2` in an isolated virtual environment (`/root/karnalm_venv`).
- **Core Frameworks:** PyTorch 2.5.1, Hugging Face `transformers` (LlamaForCausalLM), Hugging Face `datasets`, `tokenizers`, Hugging Face `trl` (SFTTrainer), `pyarrow`.
- **Telemetry & Orchestration:** GNU `tmux`, `wandb`, `huggingface_hub` API, bash process monitoring.

### System Architecture Diagram (Text Format)
```
==================================================================================================
                                    KARNALM SYSTEM ARCHITECTURE
==================================================================================================

  [ RAW DATASETS ]
  ├── FineWeb (English: 60GB / 10B tokens max)
  ├── CC-100 (Hindi: 40GB / 7B tokens max)
  ├── CC-100 / Sangraha / IndicCorp v2 (Telugu: 12GB / 5B tokens - 100% consumed)
  └── Aya & FLAN (Multilingual SFT Instructions: ~2GB)
         │
         ▼
  [ PHASE 1: TOKENIZATION & SHARDING ENGINE ]
  ├── Custom 52K Vocabulary BPE Tokenizer (trained on 1.5GB trilingual corpus slice)
  ├── 20-Worker ProcessPoolExecutor (PyArrow batch reading + NFC Normalization)
  └── Binary Sharder: Produces 19 x 1B-token uint16 shards (`train_shard_XXXX.bin`)
         │
         ▼
  [ PHASE 2: HIGH-THROUGHPUT PRETRAINING ENGINE ]
  ├── Hardware: AMD Instinct MI300X (192GB HBM3 VRAM @ 5.3 TB/s)
  ├── Environment: PyTorch 2.5.1+rocm6.2 | HSA_OVERRIDE_GFX_VERSION=9.4.2
  ├── Model: KarnaLM-360M (24 Layers, 1024 Hidden Dim, 16 Q-Heads, 8 KV-Heads / GQA)
  ├── Attention Backend: PyTorch Native SDPA (Scaled Dot Product Attention)
  ├── Precision: Pure BFloat16 | Batch Size: 16 | Grad Accum: 4 (Effective Batch: 131K tokens)
  └── Dataloader: ShardedBinaryDataset (Persistent np.memmap file descriptors)
         │
         ├──► [ Real-Time Telemetry: monitor.sh / wandb / /data/logs/training.log ]
         └──► [ Async HF Backup: ncncomplete/KarnaLM-360M-base ]
         │
         ▼
  [ PHASE 3: SUPERVISED FINE-TUNING (SFT) ENGINE ]
  ├── Framework: Hugging Face TRL (SFTTrainer)
  ├── Data: Aya + FLAN Instruction Pairs (EN, HI, TE)
  ├── Execution: Full Parameter Fine-Tuning (390M params @ lr=2e-5, Cosine Schedule)
  └── Export: Final Weights -> ncncomplete/KarnaLM-360M-chat
         │
         ▼
  [ PHASE 4: INFERENCE & BENCHMARKING ENGINE ]
  ├── Autoregressive Generation Engine (KV-Cache enabled, pure BF16)
  └── Local Verification: Benchmark Batch Sweep & Interactive Trilingual Generation
==================================================================================================
```

### Data Flow Diagram
```
+------------------+      +-----------------------+      +-------------------------+
| FineWeb (EN)     |      | CC-100 (HI)           |      | CC-100 / Sangraha (TE)  |
| Raw Parquet      |      | Raw Parquet           |      | Raw Parquet             |
+--------+---------+      +-----------+-----------+      +------------+------------+
         |                            |                               |
         +----------------------------+-------------------------------+
                                      |
                                      ▼
                   +------------------------------------+
                   | ProcessPoolExecutor (20 Workers)   |
                   | - Min length filter (>= 50 chars)  |
                   | - BPE Encoding (52K Vocab)         |
                   | - EOS Token Appending (<|endoftext|>)
                   +------------------+-----------------+
                                      |
                                      ▼
                   +------------------------------------+
                   | Capped Multilingual Buffer         |
                   | Target: 50% EN, 35% HI, 15% TE     |
                   +------------------+-----------------+
                                      |
                                      ▼
                   +------------------------------------+
                   | Sharded Binary Exporter            |
                   | uint16 array (.bin mapped)         |
                   | Output: train_shard_0000.bin (1B)  |
                   |         train_shard_0001.bin (1B)  |
                   |         ... 19 total shards        |
                   +------------------+-----------------+
                                      |
                                      ▼
                   +------------------------------------+
                   | Persistent Memmap Dataloader       |
                   | Fast Indexing into 17.8B Train     |
                   | and 50M Validation Arrays          |
                   +------------------------------------+
```

### Training Flow Diagram
```
+-----------------------------------------------------------------------------------+
| PRETRAINING STEP (03_pretrain.py)                                                  |
|                                                                                   |
|  [ ShardedBinaryDataset ] ──► [ Micro-Batch x=16x2048 ] ──► [ CUDA BF16 Cast ]     |
|                                                                    │              |
|                                                                    ▼              |
|  [ Optimizer Step (AdamW) ] ◄── [ Grad Clip (1.0) ] ◄── [ Backprop (Loss) ]        |
|               │                                                    ▲              |
|               ▼                                                    │              |
|  [ Accumulate Gradients ] ◄───────────────────────────────── [ Forward SDPA ]   |
|  (4 Micro-Steps = Effective Batch 64 = 131,072 Tokens)                            |
|                                                                                   |
|  Every 2,000 Steps: Save Local Checkpoint + Launch Async HF Upload Thread          |
+-----------------------------------------------------------------------------------+
                                      │ (70 Hours / 12.8B Tokens Completed)
                                      ▼
+-----------------------------------------------------------------------------------+
| SUPERVISED FINE-TUNING STEP (04_sft.py)                                           |
|                                                                                   |
|  [ Base Checkpoint ] ──► [ Load Aya + FLAN Instructions ] ──► [ SFTTrainer ]     |
|                                                                    │              |
|                                                                    ▼              |
|  [ Save Chat Model ] ◄── [ 2 Epochs @ LR 2e-5 ] ◄───────── [ Forward / Back ]     |
+-----------------------------------------------------------------------------------+
```

### Inference Flow Diagram
```
+-----------------------+     +-------------------------------+     +-----------------------+
| Human Prompt          | ──► | Trilingual BPE Tokenizer      | ──► | Token IDs [1024, ...] |
| (EN / HI / TE)        |     | (52,000 Vocabulary)           |     | Tensor on MI300X      |
+-----------------------+     +-------------------------------+     +-----------+-----------+
                                                                                │
                                                                                ▼
+-----------------------+     +-------------------------------+     +-----------------------+
| Output Text           | ◄── | Detokenizer                   | ◄── | Next Token ID         |
| (Streamed Token-by-T) |     | (Map IDs back to UTF-8 text)  |     | (Argmax / Sampling)   |
+-----------------------+     +-------------------------------+     +-----------+-----------+
                                                                                ▲
                                                                                │
                                      +-----------------------------------------+
                                      | Autoregressive KV-Cache Loop
                                      |
                                      |  1. Forward pass with new token
                                      |  2. Update Key-Value Cache
                                      |  3. Softmax logits over 52,000 vocab
                                      |  4. Temperature / Top-p / Top-k filtering
```

---

# Phase 2 — Repository Deep Analysis

### Folder Structure Overview

```
.
├── archive/                             # Historical notebooks, prototype scripts, and draft build plans
│   ├── KarnaLM_Build_Plan.docx          # Original Word document detailing initial 1.1B architecture strategy
│   ├── english_pipeline.ipynb           # Legacy Kaggle data processing notebook for English FineWeb
│   ├── generate_notebooks_v2.py         # Utility script to generate Kaggle execution notebooks
│   ├── hindi_pipeline.ipynb             # Legacy Kaggle data processing notebook for Hindi CC-100
│   ├── notebook1_hindi_pipeline.py      # Standalone Python conversion of Hindi processing pipeline
│   ├── notebook2_telugu_pipeline.py     # Standalone Python conversion of Telugu processing pipeline
│   ├── telugu_pipeline.ipynb            # Legacy Kaggle data processing notebook for Telugu CC-100
│   └── telugu_pipeline.py               # Standalone Python script for Telugu dataset streaming & MinHash dedup
├── dataset_final_folder/                # Kaggle-ready data pipeline notebooks and documentation
│   ├── english_pipeline.ipynb           # Final Kaggle notebook for streaming English FineWeb
│   ├── hindi_pipeline.ipynb             # Final Kaggle notebook for streaming Hindi CC-100
│   ├── instructions.md                  # Detailed Kaggle execution instructions and environment requirements
│   └── telugupipeline.ipynb             # Final Kaggle notebook for streaming Telugu Sangraha/CC-100 datasets
├── training_scripts/                    # Base training pipeline scripts (Standard CPU / local fallback)
│   ├── 00_amd_setup.sh                  # Virtual environment creation and ROCm package setup
│   ├── 01_tokenize_data.py              # Parallel BPE tokenizer training and dataset sharding script
│   ├── 02_smoke_test.py                 # Multi-point pipeline validation gatekeeper
│   ├── 03_pretrain.py                   # Main 360M pretraining loop with memmap dataloader & async HF syncing
│   ├── 04_sft.py                        # Supervised Fine-Tuning script using Hugging Face TRL SFTTrainer
│   ├── architecture.md                  # Comprehensive architectural spec sheet for initial 1.1B design
│   └── monitor.sh                       # Live terminal dashboard for tracking GPU, VRAM, tok/s, and loss
├── training_scripts_gpu/                # Active GPU-optimized deployment scripts (MI300X Production Set)
│   ├── 00_amd_setup.sh                  # MI300X specific environment initialization
│   ├── 00_verify_and_launch.sh          # One-click GPU environment test and pipeline launcher
│   ├── 01_tokenize_data.py              # 20-worker fast sharding script targeting /mnt/scratch/shards
│   ├── 02_smoke_test.py                 # MI300X smoke test with real 1.1B benchmark & batch sweep
│   ├── 03_pretrain.py                   # Production 360M pretraining script with ROCm 7.2.0 fixes & memmap
│   ├── 04_sft.py                        # Production SFT script configured for 360M final weights
│   ├── 05_train_and_deploy.sh           # End-to-end automated orchestrator (Pretrain -> HF Base -> SFT -> HF Chat -> Bench)
│   ├── architecture.md                  # Post-audit architectural log matching production 360M specs
│   ├── mini_tokenize.py                 # Fast 500MB trilingual slice extractor for 52K tokenizer training
│   └── monitor.sh                       # Production monitor script for active tmux session
├── README.md                            # Primary project landing page with setup instructions & quick progress checks
├── README_Telugu_Pipeline.md            # In-depth guide for Telugu raw dataset acquisition & MinHash LSH cleaning
├── agents.md                            # Autonomous agent execution protocol and failure troubleshooting tree
├── architectural_decisions_log.md       # Historical decision ledger tracking all major technical pivots
├── benchmark_batch.py                   # Isolated batch size sweep tool testing MI300X throughput from batch 16 to 512
├── engineering_learning_notes.md        # Technical Q&A repository covering VRAM utilization, SDPA, & next-token logic
├── history.md.resolved.5                # Historical execution log from initial environment setup phase
└── history.md.resolved.6                # Historical execution log capturing 1.1B->360M pivot and pipeline launch
```

### Folder Deep Analysis

#### 1. `training_scripts_gpu/`
- **Purpose:** Production execution suite optimized specifically for the AMD Instinct MI300X hardware instance.
- **Dependencies:** PyTorch `2.5.1+rocm6.2`, `transformers`, `tokenizers`, `pyarrow`, `datasets`, `trl`, `wandb`, `huggingface_hub`.
- **Inputs:** Raw parquet files at `/data/raw` (`/mnt/scratch/data`), HF Aya/FLAN datasets.
- **Outputs:** Binary token shards at `/mnt/scratch/shards`, model checkpoints at `/data/checkpoints/karnalm-360m`, HF repository pushes.
- **Design Decisions:** Created as a dedicated GPU folder to prevent overwriting base CPU/generic scripts while applying MI300X-specific paths (`/mnt/scratch`) and ROCm 7.2.0 compatibility flags (`HSA_OVERRIDE_GFX_VERSION=9.4.2`).

#### 2. `training_scripts/`
- **Purpose:** Portable reference implementation of the training pipeline usable on generic Linux systems or local CPU/NVIDIA environments.
- **Dependencies:** Base PyTorch, Hugging Face ecosystem.
- **Inputs:** Local `/data/raw` directory.
- **Outputs:** Local `/data/tokens` shards and `/data/checkpoints` weights.
- **Design Decisions:** Maintained standard default paths (`/data/tokens`) to ensure clean separation between generic hardware defaults and scratch-disk specific GPU configurations.

#### 3. `dataset_final_folder/`
- **Purpose:** Self-contained Kaggle execution package containing notebooks for streaming, filtering, and deduplicating raw English, Hindi, and Telugu corpora.
- **Dependencies:** Hugging Face `datasets`, `datasketch` (MinHash LSH), `sentencepiece`.
- **Inputs:** Remote Hugging Face datasets (FineWeb, CC-100, Sangraha, IndicCorp v2).
- **Outputs:** Filtered, deduplicated JSONL files uploaded directly to Hugging Face Hub (`ncncomplete/karnalm-data`).
- **Design Decisions:** Kaggle's free CPU tier offers 30GB RAM and fast bandwidth. Moving streaming deduplication to Kaggle saved GPU compute credits on the MI300X instance.

#### 4. `archive/`
- **Purpose:** Historical record of prototype scripts, draft planning documents, and early single-language processing experiments.
- **Dependencies:** Legacy project state.
- **Design Decisions:** Retained in codebase for full auditability of the project's evolution and early design iterations.

### Component Dependency Graph

```
+-------------------------------------------------------------------------------------------------+
|                                 COMPONENT DEPENDENCY GRAPH                                      |
+-------------------------------------------------------------------------------------------------+

  [ 00_amd_setup.sh ]
         │ (Creates /root/karnalm_venv & installs PyTorch 2.5.1+rocm6.2)
         ▼
  [ mini_tokenize.py ]
         │ (Extracts 500MB sample from EN/HI/TE parquet files)
         ▼
  [ 01_tokenize_data.py ]
         │ (Trains 52K BPE Tokenizer & writes train_shard_XXXX.bin to /mnt/scratch/shards)
         ▼
  [ 02_smoke_test.py ]
         │ (Validates CUDA, BF16, SDPA attention, VRAM, and benchmarks throughput)
         ▼
  [ 05_train_and_deploy.sh ] (Master Orchestrator inside tmux)
         │
         ├──► [ 03_pretrain.py ] ──► Reads Shards ──► Trains 360M Base Model
         │         │
         │         ├──► Outputs: /data/checkpoints/karnalm-360m/final
         │         └──► Triggers: Python Inline Script -> Push Base to ncncomplete/KarnaLM-360M-base
         │
         ├──► [ 04_sft.py ] ──► Loads Base Model + Aya/FLAN ──► Fine-tunes Assistant
         │         │
         │         ├──► Outputs: /data/checkpoints/karnalm-360m-chat
         │         └──► Triggers: Python Inline Script -> Push Chat to ncncomplete/KarnaLM-360M-chat
         │
         └──► Local Inference Benchmark ──► Verifies BF16 text generation
+-------------------------------------------------------------------------------------------------+
```

### Module Interaction Map

```
+--------------------------+         +-------------------------------+         +----------------------------+
| 01_tokenize_data.py      |         | ShardedBinaryDataset          |         | LlamaForCausalLM           |
| - Reads Raw Parquet      | ──────► | - np.memmap (uint16)          | ──────► | - Hugging Face Model       |
| - Custom 52K BPE         |         | - Fast Slice [idx:idx+2048]   |         | - SDPA Attention Backend   |
+--------------------------+         +-------------------------------+         +-------------+--------------+
                                                                                             │
                                                                                             ▼
+--------------------------+         +-------------------------------+         +----------------------------+
| HfApi (huggingface_hub)  |         | Background Upload Thread      |         | AdamW Optimizer            |
| - Async Checkpoint Sync  | ◄────── | - Non-blocking thread launch  | ◄────── | - Loss.backward()          |
| - Model Hub Uploads      |         | - Syncs every 2,000 steps     |         | - Step & zero_grad         |
+--------------------------+         +-------------------------------+         +----------------------------+
```

---

# Phase 3 — Model Architecture Audit

### Model Specification Sheet

| Attribute | Initial Design (1.1B) | Production Architecture (KarnaLM-360M) |
| :--- | :--- | :--- |
| **Base Architecture** | LLaMA-style Decoder-Only | LLaMA-style Decoder-Only |
| **Total Parameters** | 1,100,000,000 (1.1B) | **390,123,520 (~390M Total / 337M Core + 53M Embeddings)** |
| **Hidden Dimension ($d_{model}$)** | 2,048 | **1,024** |
| **Number of Layers ($N_L$)** | 22 | **24** |
| **Attention Heads ($N_H$)** | 32 | **16** (Head Dim = $1024 / 16 = 64$) |
| **Key-Value Heads ($N_{KV}$)** | 8 (Grouped Query Attention) | **8** (Grouped Query Attention — 2:1 Q to KV ratio) |
| **Intermediate Size ($d_{ff}$)** | 5,632 | **2,816** (SwiGLU Activation) |
| **Vocabulary Size ($V$)** | 52,000 | **52,000** |
| **Max Context Window** | 2,048 tokens | **2,048 tokens** |
| **Embedding Dimension** | $52,000 	imes 1,024$ | **$52,000 	imes 1,024 = 53,248,000$ parameters** |
| **Positional Encoding** | RoPE ($	heta = 10000.0$) | **RoPE ($	heta = 10000.0$)** |
| **Normalization** | RMSNorm ($\epsilon = 1e-5$) | **RMSNorm ($\epsilon = 1e-5$)** |
| **Attention Backend** | Flash Attention 2 (Attempted) | **PyTorch Native SDPA (`attn_implementation="sdpa"`)** |
| **Tie Word Embeddings** | False | **False** |

### Parameter Breakdown Calculation
1. **Embedding Layer:** $V 	imes d_{model} = 52,000 	imes 1,024 = 53,248,000$
2. **Per Transformer Layer:**
   - **Query Projection ($W_q$):** $d_{model} 	imes (N_H 	imes d_{head}) = 1,024 	imes 1,024 = 1,048,576$
   - **Key Projection ($W_k$):** $d_{model} 	imes (N_{KV} 	imes d_{head}) = 1,024 	imes (8 	imes 64) = 524,288$
   - **Value Projection ($W_v$):** $d_{model} 	imes (N_{KV} 	imes d_{head}) = 1,024 	imes (8 	imes 64) = 524,288$
   - **Output Projection ($W_o$):** $(N_H 	imes d_{head}) 	imes d_{model} = 1,024 	imes 1,024 = 1,048,576$
   - **Attention Total:** $1,048,576 + 524,288 + 524,288 + 1,048,576 = 3,145,728$
   - **SwiGLU FFN ($W_{gate}, W_{up}, W_{down}$):** $3 	imes (d_{model} 	imes d_{ff}) = 3 	imes (1,024 	imes 2,816) = 8,650,752$
   - **RMSNorms ($2 	imes d_{model}$):** $2 	imes 1,024 = 2,048$
   - **Total per Layer:** $3,145,728 + 8,650,752 + 2,048 = 11,798,528$
3. **24 Layers Core Total:** $24 	imes 11,798,528 = 283,164,672$
4. **Final Output Head ($W_{head}$):** $d_{model} 	imes V = 1,024 	imes 52,000 = 53,248,000$
5. **Grand Total:** $53,248,000 + 283,164,672 + 53,248,000 = 389,660,672$ (~390M Parameters).

### Why This Architecture Was Selected & The 1.1B Pivot
The original project roadmap called for a 1.1B parameter LLaMA model (22 Layers, 2048 Dim, 32 Heads). However, during hardware benchmarking (`02_smoke_test.py` and `benchmark_batch.py`), real-world MI300X execution revealed a critical computational ceiling:
- **1.1B Benchmark Speed:** At safe batch sizes (Batch 4/8), the 1.1B model achieved a peak throughput of **~22,000 tokens/second**.
- **Time/Budget Math:** Pretraining a 1.1B model on the 17.8B token corpus at 22,000 tok/s required **224 to 244 hours**. With a total compute credit budget capped at **$190 (~95 hours)**, training a 1.1B model would exhaust funds at ~18% completion, yielding an severely undertrained, useless model.
- **The Chinchilla Solution:** Chinchilla scaling laws ($N pprox 20D$) state that a model is optimally trained when it sees approximately 20 tokens per parameter. For a 360M parameter model, optimal convergence occurs at **~7.2B to 12.8B tokens**.
- **360M Benchmark Speed:** Rescaling the architecture down to 360M (24 Layers, 1024 Dim, 16 Heads) skyrocketed computational throughput to **~51,500 tokens/second**.
- **Final Pretraining Horizon:** At 51,500 tok/s, processing **12.8 Billion tokens** required exactly **68.9 hours**, perfectly fitting within the compute budget while leaving a 25-hour safety buffer for SFT, deployment, and evaluation. A fully converged 360M model trained on 12.8B tokens vastly outperforms an undertrained 1.1B model.

---

# Phase 4 — Tokenizer Audit

### Tokenizer Specifications
- **Type:** Byte-Pair Encoding (BPE) via Hugging Face `tokenizers.Tokenizer`.
- **Vocabulary Size:** 52,000 tokens.
- **Normalization:** NFC (Normalization Form C) unicode normalization.
- **Pre-tokenizer:** ByteLevel pre-tokenizer (handling raw byte fallbacks to prevent out-of-vocabulary `<unk>` tokens).
- **Special Tokens:**
  - `<pad>` (ID: 0) — Padding token
  - `<unk>` (ID: 1) — Unknown token fallback
  - `<|endoftext|>` (ID: 2) — Sequence boundary / EOS token
  - `<s>` (ID: 3) — Start of sequence
  - `</s>` (ID: 4) — End of sequence
- **Special Token Enforcement:** `EOS_ID = 2` explicitly appended to every encoded document during binary dataset sharding.

### How Text Becomes Tokens
1. **Unicode Normalization:** NFC standardizes visual character variations (e.g., combining Devanagari/Telugu accents).
2. **Byte-Level Pre-tokenization:** Splitting text along whitespace and punctuation while preserving character boundaries using byte representation.
3. **BPE Merging:** Iterative merging of most frequent byte pairs up to 52,000 merges based on statistics gathered from the 1.5GB trilingual calibration corpus (`mini_tokenize.py`).

### Fertility Analysis
Token fertility is defined as the average number of tokens generated per word ($	ext{Fertility} = rac{	ext{Token Count}}{	ext{Word Count}}$).

| Language | Standard LLaMA Vocabulary (32K) | KarnaLM Custom Vocabulary (52K) | Efficiency Gain |
| :--- | :--- | :--- | :--- |
| **English** | ~1.3 tokens/word | **1.2 tokens/word** | ~8% faster |
| **Hindi** | ~6.5 tokens/word | **2.4 tokens/word** | **2.7x faster** |
| **Telugu** | ~15.0 tokens/word | **3.9 tokens/word** | **3.8x faster** |

---

# Phase 5 — Dataset Investigation

### Data Ecosystem & Mix

| Source Corpus | Target Language | Raw File Format | Raw Disk Size | Filtered Tokens | Target Pretraining Mix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FineWeb (Hugging Face)** | English | Parquet | 60 GB | ~10.0 Billion | **50.0% (6.4B tokens)** |
| **CC-100 Hindi** | Hindi | Parquet | 40 GB | ~7.0 Billion | **35.0% (4.5B tokens)** |
| **CC-100 / Sangraha / IndicCorp v2** | Telugu | Parquet | 12 GB | ~1.82 Billion | **15.0% (1.82B tokens - 100% consumed)** |
| **Aya & FLAN Datasets** | Instruction Pairs | JSONL | 2 GB | SFT Phase | **Supervised Fine-Tuning** |
| **Total** | **Trilingual** | **Parquet / Bin** | **114 GB** | **18.82 Billion** | **12.8B Training Target** |

### Data Cleaning & Deduplication Pipeline
1. **Document Filtering (`dataset_final_folder/`):** Documents shorter than 50 characters or containing non-text junk were stripped during streaming.
2. **Script Validation:** Filtered Telugu streams required >30% of alphabetic characters to lie within the Unicode Telugu range (`U+0C00` to `U+0C7F`).
3. **MinHash LSH Deduplication:** Applied in single-pass streaming mode using `datasketch.MinHashLSH` (threshold 0.85 Jaccard similarity) to remove duplicate web scrapes across CC-100 and OSCAR.
4. **Hard Capped Sampling Strategy:** To prevent English from overwhelming Telugu, hard caps were set during tokenization:
   - English Cap: 10B tokens
   - Hindi Cap: 7B tokens
   - Telugu Cap: 5B tokens (Exceeded total available Telugu data volume, forcing **100% of the Telugu corpus** into the binary shards).

---

# Phase 6 — Training Pipeline Reverse Engineering

### Pretraining Hyperparameters (`03_pretrain.py`)

| Parameter | Setting | Engineering Justification |
| :--- | :--- | :--- |
| **Training Framework** | Custom PyTorch + Hugging Face Transformers | Clean control over execution loops & memmap dataloading |
| **Precision Mode** | Pure **BFloat16** (`torch.bfloat16`) | Native hardware acceleration on MI300X; eliminates FP16 underflow/overflow scaling issues |
| **Micro Batch Size** | 16 sequences per step | Maximizes matrix compute unit occupancy without triggering VRAM fragmentation |
| **Gradient Accumulation** | 4 steps | Yields an effective batch size of $16 	imes 4 = 64$ sequences |
| **Effective Batch Size** | **64 sequences (131,072 tokens/step)** | Standard optimal batch size for ~300M-500M parameter models |
| **Peak Learning Rate** | $5.0 	imes 10^{-4}$ ($5e-4$) | Standard peak rate for 300M LLaMA models |
| **Min Learning Rate** | $5.0 	imes 10^{-5}$ ($5e-5$) | $10\%$ decay floor for stable final convergence |
| **LR Scheduler** | Cosine Annealing with Warmup | Smooth initial gradient trajectory followed by optimal convergence decay |
| **Warmup Steps** | 2,000 steps (~262M tokens) | Prevents initial large gradients from destabilizing untrained embeddings |
| **Optimizer** | AdamW ($eta_1=0.9, eta_2=0.95, \epsilon=1e-8$) | LLaMA standard momentum configuration |
| **Weight Decay** | 0.1 | Prevents parameter magnitude blowup during long pretraining |
| **Gradient Clipping** | 1.0 (Max Norm) | Prevents catastrophic gradient spikes during pretraining |
| **Checkpoint Interval** | Every 2,000 steps (~262M tokens) | Ensures minimal lost compute in case of instance preemption |

---

# Phase 7 — AMD GPU Investigation & ROCm Engineering

### Hardware & Driver Specification
- **GPU Model:** AMD Instinct MI300X (Single OAM Module)
- **Architecture:** CDNA3 (`gfx942`)
- **VRAM Capacity:** 192GB HBM3
- **Memory Bandwidth:** **5.3 TB/s** (vs 3.35 TB/s on NVIDIA H100 SXM5)
- **Host System Runtime Driver:** AMD ROCm 7.2.0

### ROCm Compatibility Audit & Resolution
During project setup, the initial execution environment encountered severe execution failures:
1. **The ROCm 7.2.0 C++ ABI Mismatch:** The host system driver was running ROCm 7.2.0, while pre-installed PyTorch binaries were compiled for ROCm 6.1. This mismatch caused fatal `RuntimeError: HIP error: invalid device function` and `Segmentation fault` crashes during basic matrix multiplication tests.
2. **The PEP 668 PEP Block:** Global pip installs were blocked by Ubuntu's externally managed environment setting.
3. **The Solution:** An isolated virtual environment was created at `/root/karnalm_venv`. PyTorch was updated to `2.5.1+rocm6.2` via AMD wheels. Environmental flags were configured:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=9.4.2
   export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
   ```
4. **Flash Attention vs PyTorch SDPA:** Building native `flash-attn` from source failed on ROCm due to compiler header mismatches. The pipeline pivoted to PyTorch's native **Scaled Dot Product Attention (SDPA)** (`attn_implementation="sdpa"`), which automatically invokes pre-compiled ROCm C++ kernels without compilation overhead.

---

# Phase 8 — Fine-Tuning Analysis (SFT)

### Fine-Tuning Strategy (`04_sft.py`)
- **Methodology:** **Full Parameter Fine-Tuning** (all 390M parameters unfrozen).
- **Framework:** Hugging Face `trl.SFTTrainer`.
- **Dataset:** Combined Aya Dataset + FLAN Multilingual Instruction Pairs in English, Hindi, and Telugu (~100,000 instruction-response pairs).
- **Max Sequence Length:** 2,048 tokens.
- **Learning Rate:** $2.0 	imes 10^{-5}$ ($2e-5$) — 25x smaller than pretraining peak rate ($5e-4$).
- **Epochs:** 2 epochs.
- **Effective Batch Size:** 64 sequences (Micro batch 16, Grad accum 4).
- **Execution Time:** ~12 hours on AMD MI300X.

---

# Phase 9 — Optimization Audit

1. **PyTorch Native SDPA:** Replaced uncompilable native C++ Flash Attention with PyTorch native SDPA (`attn_implementation="sdpa"`), achieving 51,500 tok/s with zero compilation risk.
2. **Persistent Memmap Dataloader (`ShardedBinaryDataset`):** Initial dataloader implementations opened new `np.memmap` handles per micro-batch, exhausting Linux file descriptors at 51,500 tok/s. The dataloader was rewritten to open all 19 binary shards once during initialization and maintain persistent mapped memory arrays.
3. **Pure BFloat16 Training:** Avoided FP16 loss scaling overhead while maintaining numerical stability across all 24 layers.
4. **Grouped Query Attention (GQA):** 8 KV heads serving 16 Q heads reduced KV cache memory bandwidth pressure during evaluation and serving by 50%.
5. **Multi-Worker PyArrow Sharding (`01_tokenize_data.py`):** 20 worker parallel processing converted 114GB of raw parquet files into binary tokens in under 90 minutes.

---

# Phase 10 — Experiment History Reconstruction

### Execution Timeline
1. **ROCm Matrix Failure:** Attempted initial run under ROCm 6.1 PyTorch binaries -> Encountered `invalid device function` crash -> Resolved by installing `torch-2.5.1+rocm6.2` in isolated venv with `HSA_OVERRIDE_GFX_VERSION=9.4.2`.
2. **Flash Attention Build Wall:** Attempted native C++ `flash-attn` compilation -> Build failed on AMD headers -> Pivoted to PyTorch native SDPA.
3. **Tokenizer Calibration:** Extracted 500MB sample per language -> Trained 52K BPE tokenizer -> Verified Telugu fertility drop from 15.0 to 3.9 tokens/word.
4. **The 1.1B -> 360M Budget Pivot:** Ran smoke test on 1.1B architecture -> Measured throughput ceiling of 22,000 tok/s -> Calculated 244-hour training time exceeding $190 / 95-hour budget -> Pivoted to Chinchilla-optimal 360M parameter model achieving 51,500 tok/s (70h pretraining time).
5. **Dataloader File Descriptor Leak Fix:** Discovered open file handle leak in memmap dataloader -> Rewrote `ShardedBinaryDataset` to maintain persistent file descriptors.
6. **Pipeline Automation:** Created `05_train_and_deploy.sh` script executing in detached `tmux` session to automate Pretraining -> Base Model Push -> SFT -> Chat Model Push -> Benchmarking.
---

# Phase 11 — Complete Code Walkthrough

This section provides an exhaustive line-by-line, multi-tier walkthrough of all primary training and deployment functions in the repository.

### 1. Parallel Tokenization & Binary Exporter (`01_tokenize_data.py`)

#### Function: `process_single_parquet(fpath, tok_path, text_col, eos_id)`

```python
def process_single_parquet(fpath: Path, tok_path: Path, text_col: str, eos_id: int):
    tok = Tokenizer.from_file(str(tok_path))
    lang = detect_language_from_path(fpath)
    results = []
    n_docs = 0
    n_tokens = 0

    try:
        parquet_file = pq.ParquetFile(fpath)
        for batch in parquet_file.iter_batches(batch_size=1000, columns=[text_col]):
            for row in batch[text_col]:
                text = str(row).strip()
                if len(text) < 50:
                    continue
                ids = tok.encode(text).ids
                ids.append(eos_id)
                results.append(np.array(ids, dtype=np.uint16))
                n_tokens += len(ids)
                n_docs += 1
    except Exception as e:
        log.error(f"Error reading {fpath}: {e}")

    return (lang, results, n_docs, n_tokens, str(fpath))
```

- **Line-by-line Behavior:**
  - `tok = Tokenizer.from_file(str(tok_path))`: Loads the compiled 52K vocabulary BPE tokenizer instance per worker process.
  - `lang = detect_language_from_path(fpath)`: Infers whether the shard belongs to English, Hindi, or Telugu based on folder taxonomy (`/en/`, `/hi/`, `/te/`).
  - `parquet_file.iter_batches(batch_size=1000, columns=[text_col])`: Reads raw parquet files in memory-bounded chunks of 1,000 documents using Apache Arrow to prevent CPU RAM overflow.
  - `if len(text) < 50: continue`: Discards trivial whitespace, empty documents, or corrupted short strings.
  - `ids = tok.encode(text).ids`: Executes BPE byte-pair encoding algorithm on unicode NFC string.
  - `ids.append(eos_id)`: Appends special sequence boundary token `<|endoftext|>` (`ID = 2`).
  - `results.append(np.array(ids, dtype=np.uint16))`: Converts Python list to memory-compact 16-bit unsigned integer array.
- **Inputs:** File path (`Path`), Tokenizer location (`Path`), Text column name (`str`), EOS Token ID (`int`).
- **Outputs:** Tuple containing language label, list of 16-bit token numpy arrays, document count, token count, and file path.
- **Mathematical & ML Purpose:** Converts unstructured text strings $S$ into discrete categorical token indices $X = (x_1, x_2, \dots, x_N) \in \{0, \dots, V-1\}^N$. Explicitly appends EOS tokens to define causal domain boundaries for next-token prediction loss calculations.
- **Explanation Tiers:**
  - **Beginner:** This function opens a single dataset file, reads words, converts them into numbers using our word-dictionary, adds a "stop" signal at the end of each document, and packs them tightly into binary memory.
  - **Intermediate:** It uses PyArrow batching to stream parquet tables in 1000-row chunks, filters out low-quality short texts (<50 characters), encodes text with custom 52K BPE, appends EOS token 2, and casts indices to `uint16` to halve RAM footprint.
  - **Expert:** Parallel worker process execution avoiding Python's GIL. Memory footprint is strictly bounded to $O(	ext{batch\_size} 	imes L_{avg})$. Output `uint16` conversion compresses token representations from 64-bit Python integers to 2 bytes per token, matching the maximum vocabulary size boundary $V = 52,000 \le 2^{16}-1 = 65,535$.

---

### 2. Pipeline Gatekeeper & Benchmark Suite (`02_smoke_test.py`)

#### Function: `run_hardware_and_throughput_benchmark()`

```python
def run_hardware_and_throughput_benchmark():
    check("CUDA available", torch.cuda.is_available())
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check("VRAM > 150GB", vram > 150, f"{vram:.0f} GB detected")
    check("BF16 supported", torch.cuda.is_bf16_supported())

    # Instantiate 1.1B model benchmark
    config_1b = LlamaConfig(
        hidden_size=2048, num_hidden_layers=22,
        num_attention_heads=32, num_key_value_heads=8,
        intermediate_size=5632, vocab_size=52000,
        attn_implementation="sdpa"
    )
    model = LlamaForCausalLM(config_1b).to("cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run 10 warm-up and timed steps at Batch 4 / Seq 2048
    x = torch.randint(0, 52000, (4, 2048), device="cuda")
    t0 = time.time()
    for _ in range(10):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = (10 * 4 * 2048) / elapsed
    print(f"1.1B Throughput: {tps:,.0f} tok/sec")
```

- **Line-by-line Behavior:**
  - Queries ROCm HIP device properties via PyTorch CUDA interface.
  - Builds full 1.1B LLaMA configuration in VRAM.
  - Allocates synthetic dummy batch $X \in \mathbb{Z}^{4 	imes 2048}$ with uniform random integers in range $[0, 52000)$.
  - Executes mixed precision forward pass, backward gradient pass, optimizer step, and zero-gradient clearing.
  - Issues `torch.cuda.synchronize()` to flush the asynchronous ROCm HIP execution command queue before computing elapsed time.
  - Calculates tokens per second $	ext{TPS} = rac{N_{steps} 	imes B 	imes T}{\Delta t}$.
- **Inputs:** Hardware environment state and PyTorch configuration objects.
- **Outputs:** Booleans for gatekeeper checks, measured VRAM allocation, and measured tok/s throughput.
- **Mathematical & ML Purpose:** Empirically measures FLOPS and memory bandwidth limits of the GPU under CDNA3 matrix instruction execution without incurring dataset loader noise.
- **Explanation Tiers:**
  - **Beginner:** Tests if the GPU is working, loads a sample model into memory, runs 10 test training steps, and measures how many words per second the GPU can train.
  - **Intermediate:** It checks ROCm driver health, allocates a 1.1B parameter LLaMA model in BFloat16, executes 10 iterations of forward-backward passes with AdamW, and calculates real-world token throughput to verify project timeline viability.
  - **Expert:** Evaluates CDNA3 matrix core throughput under PyTorch SDPA. Hardware synchronization via `hipDeviceSynchronize` ensures exact kernel execution timing, revealing the 22,000 tok/s speed bottleneck that triggered the pivot to 360M.

---

### 3. Production Pretraining Loop (`03_pretrain.py`)

#### Class: `ShardedBinaryDataset` & Training Loop Step

```python
class ShardedBinaryDataset(Dataset):
    def __init__(self, shard_paths, seq_len=2048):
        self.seq_len = seq_len
        self.shards = [np.memmap(p, dtype=np.uint16, mode='r') for p in shard_paths]
        self.shard_lengths = [(len(s) - 1) // seq_len for s in self.shards]
        self.cumulative_lengths = np.cumsum(self.shard_lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        shard_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]
        start_pos = local_idx * self.seq_len
        chunk = self.shards[shard_idx][start_pos : start_pos + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])
```

- **Line-by-line Behavior:**
  - `self.shards = [np.memmap(...)]`: Maps all binary shard files into kernel page memory without reading whole files into RAM.
  - `np.searchsorted(...)`: Performs binary search $O(\log S)$ over cumulative shard boundary array to locate target shard for sequence index `idx`.
  - `chunk[:-1]` and `chunk[1:]`: Slices input sequence $X = (x_0, x_1, \dots, x_{T-1})$ and ground-truth autoregressive target labels $Y = (x_1, x_2, \dots, x_T)$.
- **Inputs:** Shard file paths list (`list[str]`), Sequence length (`int = 2048`), Sequence index (`int`).
- **Outputs:** Tuple of PyTorch Tensors `(input_ids, labels)` of shape `(2048,)` and dtype `torch.int64`.
- **Mathematical & ML Purpose:** Generates autoregressive input-target pairs for causal language modeling. Inputs $X$ predict targets $Y$ shifted by 1 position.
- **Explanation Tiers:**
  - **Beginner:** A smart data reader that opens 19 massive data files instantly without crashing system memory, grabbing 2048 words for the model to read and 2048 target words for the model to predict.
  - **Intermediate:** Uses virtual memory mapping (`np.memmap`) to keep memory footprint flat (<10% host CPU RAM). Applies binary search over cumulative shard length arrays to achieve $O(\log S)$ slice retrieval.
  - **Expert:** Retains persistent OS page cache handles, preventing file descriptor leaks across millions of micro-batches. Ensures zero-copy CPU-to-GPU memory transfer via page-locked pinned memory tensors.

---

# Phase 12 — Mathematical Foundations

This section provides rigorous mathematical derivations, intuitive explanations, interview definitions, and follow-up Q&As for every foundational algorithm used in KarnaLM.

---

### 1. Self-Attention & Scaled Dot-Product Attention

#### Mathematical Derivation
Given input sequence matrix $X \in \mathbb{R}^{T 	imes d_{model}}$, linear projections compute Queries, Keys, and Values:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad 	ext{where } W_Q, W_K, W_V \in \mathbb{R}^{d_{model} 	imes d_k}$$
The scaled dot-product attention map is computed as:
$$A = 	ext{softmax}\left( rac{Q K^T}{\sqrt{d_k}} + M ight) \in \mathbb{R}^{T 	imes T}$$
$$	ext{Attention}(Q, K, V) = A V \in \mathbb{R}^{T 	imes d_k}$$
Where $M$ is the causal mask matrix:
$$M_{i,j} = egin{cases} 0 & 	ext{if } i \ge j \ -\infty & 	ext{if } i < j \end{cases}$$

#### Intuition
Attention allows every token in a sequence to dynamically dynamic weight and aggregate information from every preceding token. The scaling factor $rac{1}{\sqrt{d_k}}$ prevents dot products from growing excessively large in higher dimensions, which would drive Softmax gradients into vanishing regions.

#### Interview Explanation
"Scaled Dot-Product Attention measures pairwise token affinity by taking the dot product of Query and Key vectors. We scale by $\sqrt{d_k}$ to maintain unit variance before Softmax, preventing vanishing gradients, and apply a lower-triangular causal mask to enforce autoregressive token generation."

#### Common Follow-Up Questions & Answers
- **Q:** Why do we scale by $\sqrt{d_k}$ specifically?
  - **A:** Assuming $q_i, k_i \sim \mathcal{N}(0, 1)$ are independent random variables, their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$. Dividing by $\sqrt{d_k}$ renormalizes the variance back to $1$, keeping Softmax input magnitudes in a stable gradient regime.

---

### 2. Grouped Query Attention (GQA)

#### Mathematical Formula
Let $N_H$ be the number of Query heads and $N_{KV}$ be the number of Key-Value heads, where $N_{KV} = N_H / G$. Queries are grouped into $G$ groups:
$$	ext{Group}_g = \{ Q_{g \cdot G + 1}, Q_{g \cdot G + 2}, \dots, Q_{(g+1) \cdot G} \}$$
Each query in $	ext{Group}_g$ attends to the shared key-value head pair $(K_g, V_g)$:
$$	ext{Head}_{i} = 	ext{Attention}\left(Q_i, K_{\lfloor i / G floor}, V_{\lfloor i / G floor}ight)$$

#### Intuition
Multi-Head Attention (MHA) allocates separate KV heads for every Query head ($N_{KV} = N_H$), which inflates KV cache size during generation. Multi-Query Attention (MQA) shares a single KV head across all Query heads ($N_{KV} = 1$), which can degrade model quality. GQA provides the middle ground by grouping Queries (e.g., $N_H=16, N_{KV}=8$, group size $G=2$), achieving 2x KV memory savings with MHA-level perplexity.

#### Interview Explanation
"GQA divides Query heads into groups that share Key and Value heads. In KarnaLM, 16 Query heads share 8 KV heads, cutting KV-cache memory bandwidth consumption by 50% during autoregressive generation with no loss in model capacity."

---

### 3. SwiGLU Activation Function

#### Mathematical Formula
$$	ext{Swish}_eta(x) = x \cdot \sigma(eta x) = rac{x}{1 + e^{-eta x}}$$
$$	ext{SwiGLU}(x, W_{gate}, W_{up}, W_{down}) = \left( 	ext{Swish}_1(x W_{gate}) \odot (x W_{up}) ight) W_{down}$$

#### Intuition
Gated Linear Units (GLUs) control information flow by multiplying two linear transformations, one of which acts as a continuous non-linear gate. Using Swish as the gating function provides smooth, non-monotonic gradient flow across negative activation values.

#### Interview Explanation
"SwiGLU replaces standard GELU/ReLU in the Transformer Feed-Forward Network. It computes an element-wise product between a Swish-gated linear transformation and a secondary linear projection. To maintain parameter equivalence with standard $4 d_{model}$ FFNs, we set $d_{ff} pprox rac{8}{3} d_{model} = 2816$."

---

### 4. Root Mean Square Normalization (RMSNorm)

#### Mathematical Formula
$$	ext{RMS}(x) = \sqrt{rac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$
$$	ext{RMSNorm}(x)_i = rac{x_i}{	ext{RMS}(x)} \odot \gamma_i$$

#### Intuition
LayerNorm computes both mean and variance $rac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$. Research shows that the scaling property (variance normalization) is what drives training stability, not mean centering. RMSNorm removes mean subtraction entirely, saving $2d$ arithmetic operations per layer.

#### Interview Explanation
"RMSNorm normalizes activations by their root mean square rather than full variance and mean. It offers identical stability to LayerNorm while reducing computational overhead by 10-50% on hardware kernels."

---

### 5. Rotary Position Embeddings (RoPE)

#### Mathematical Formula
Given 2D vector component $(x_1, x_2)^T$ at sequence position $m$:
$$R_{\Theta, m}^{(2i, 2i+1)} egin{pmatrix} x_{2i} \ x_{2i+1} \end{pmatrix} = egin{pmatrix} \cos m	heta_i & -\sin m	heta_i \ \sin m	heta_i & \cos m	heta_i \end{pmatrix} egin{pmatrix} x_{2i} \ x_{2i+1} \end{pmatrix}$$
Where $	heta_i = 10000^{-2(i-1)/d}$.

#### Intuition
Rather than adding absolute position vectors to embeddings at the input layer, RoPE multiplies Query and Key vectors by a position-dependent rotation matrix in the complex plane. The dot product between rotated Query at position $m$ and Key at position $n$ depends purely on the relative distance $m - n$.

#### Interview Explanation
"RoPE encodes relative positional information directly into attention dot products by rotating Query and Key vectors by an angle proportional to their absolute sequence position. It generalizes cleanly to unseen sequence lengths."

---

### 6. AdamW Optimizer Mathematics

#### Mathematical Formula
At step $t$, with objective gradient $g_t =
abla_	heta \mathcal{L}(	heta_t)$:
$$m_t = eta_1 m_{t-1} + (1 - eta_1) g_t \quad 	ext{(1st Moment Vector)}$$
$$v_t = eta_2 v_{t-1} + (1 - eta_2) g_t^2 \quad 	ext{(2nd Uncentered Moment Vector)}$$
$$\hat{m}_t = rac{m_t}{1 - eta_1^t}, \quad \hat{v}_t = rac{v_t}{1 - eta_2^t} \quad 	ext{(Bias Corrections)}$$
$$	heta_{t+1} = 	heta_t - \eta_t \left( rac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda 	heta_t ight) \quad 	ext{(Weight Decay Decoupled)}$$

#### Intuition
Standard Adam applies weight decay to the gradient, which mixes weight penalty with adaptive momentum moments. AdamW decouples weight decay by subtracting $\eta_t \lambda 	heta_t$ directly from the weights after computing the adaptive step.

---

### 7. Cross-Entropy Loss & Autoregressive Pretraining

#### Mathematical Formula
$$\mathcal{L}_{CE} = -rac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, \dots, x_{t-1})$$
$$P(x_t = k \mid x_{<t}) = rac{e^{z_{t,k}}}{\sum_{j=1}^{V} e^{z_{t,j}}}$$

#### Intuition
Measures the negative log-likelihood of predicting the true target token $x_t$ given all historical context $x_{<t}$. Minimizing cross-entropy minimizes the KL-divergence between the model's predicted output distribution and the true data distribution.

---

# Phase 13 — Production & Inference Audit

### Model Loading & KV-Cache Management
During inference, loading model weights into VRAM consumes $390 	imes 10^6 	imes 2 	ext{ bytes} pprox 780	ext{ MB}$ in BF16 precision. Autoregressive token generation generates one token at a time. Without Key-Value caching, generating token $t$ requires recomputing $Q, K, V$ projections for all historical tokens $1 \dots t-1$, resulting in $O(t^2)$ sequence generation complexity.

KarnaLM's inference engine enables `use_cache=True`. During token generation step $t$:
1. $K_t, V_t$ are computed only for the newly appended prompt/generated token.
2. $K_t, V_t$ are concatenated onto the persistent KV-Cache tensor $K_{1:t-1}, V_{1:t-1}$ stored in VRAM.
3. Attention is computed between $Q_t$ and cached $K_{1:t}, V_{1:t}$, reducing per-token generation complexity from $O(t^2)$ to $O(t)$.

### Sampling Methods & Parameters

| Parameter | Recommended Setting | Mathematical Operation | ML Effect |
| :--- | :--- | :--- | :--- |
| **Temperature ($T$)** | `0.7` | $	ilde{z}_i = z_i / T$ | Scales logit sharpness. $T < 1.0$ increases probability of peak tokens, reducing gibberish and hallucination. |
| **Top-$k$** | `50` | Keep top $k$ highest logits; set rest to $-\infty$. | Hard truncation of improbable tail vocabulary tokens. |
| **Top-$p$ (Nucleus)** | `0.9` | Keep smallest subset where $\sum P(x) \ge p$. | Dynamic thresholding based on cumulative probability mass. |
| **Repetition Penalty**| `1.1` | $z_i = z_i / 	heta$ if token $i$ appeared in prompt. | Penalizes duplicate n-gram loops during long context generation. |

---

# Phase 14 — Comprehensive Interview Preparation Bank

This section contains complete, expert-level answers for questions across all 7 interview categories.

---

## 1. Beginner Questions (50 Questions & Detailed Answers)

#### Q1: What is an LLM?
**Answer:** A Large Language Model (LLM) is a deep neural network, typically based on the Transformer architecture, trained on vast amounts of text data using self-supervised learning (usually next-token prediction) to understand and generate human-like text.

#### Q2: What does autoregressive language modeling mean?
**Answer:** Autoregressive language modeling means the model predicts the next token in a sequence conditioned on all previously generated tokens, feeding its own previous outputs back into the model as inputs for subsequent steps.

#### Q3: What is a token?
**Answer:** A token is a fundamental unit of text processed by a language model. It can represent a word, subword, character, or byte sequence depending on the tokenizer dictionary.

#### Q4: What is vocabulary size ($V$)?
**Answer:** Vocabulary size is the total number of unique tokens defined in the model's tokenizer dictionary. KarnaLM uses a vocabulary size of 52,000 tokens.

#### Q5: What is BPE (Byte-Pair Encoding)?
**Answer:** BPE is a subword tokenization algorithm that begins with individual characters/bytes and iteratively merges the most frequently co-occurring pairs of tokens in a training corpus until a target vocabulary size is reached.

#### Q6: What is token fertility?
**Answer:** Token fertility is the average ratio of tokens generated per word ($	ext{Tokens} / 	ext{Words}$). Lower fertility means higher tokenization efficiency.

#### Q7: Why is high fertility bad for Indic languages?
**Answer:** High fertility splits words into dozens of fragment tokens, inflating sequence length, destroying the effective context window, and increasing training and generation costs by 3-4x.

#### Q8: What language corpus mix does KarnaLM use?
**Answer:** KarnaLM uses a trilingual mix of 50% English (FineWeb), 35% Hindi (CC-100), and 15% Telugu (CC-100/Sangraha/IndicCorp v2).

#### Q9: What is pretraining?
**Answer:** Pretraining is the first stage of LLM training where the model learns general language structure, facts, grammar, and reasoning by predicting missing/next tokens on billions of unlabelled text documents.

#### Q10: What is Supervised Fine-Tuning (SFT)?
**Answer:** SFT is the second stage of training where a pretrained base model is trained on prompt-response instruction pairs to teach it to act as a helpful conversational assistant.

#### Q11: What is context window length?
**Answer:** Context window length is the maximum number of tokens a model can process in a single forward pass. KarnaLM's context window is 2,048 tokens.

#### Q12: What is the parameter count of KarnaLM-360M?
**Answer:** KarnaLM-360M has approximately 390 million total parameters (337M core transformer layers + 53M embedding weights).

#### Q13: What GPU was used to train KarnaLM?
**Answer:** AMD Instinct MI300X with 192GB HBM3 VRAM.

#### Q14: What is ROCm?
**Answer:** ROCm (Radeon Open Compute) is AMD's open-source software platform and driver architecture for GPU computing, parallel to NVIDIA CUDA.

#### Q15: What precision format was used during training?
**Answer:** Pure BFloat16 (`torch.bfloat16`).

#### Q16: What is BFloat16?
**Answer:** Brain Floating Point 16 is a 16-bit numerical format with 1 sign bit, 8 exponent bits, and 7 mantissa bits, offering the same dynamic exponent range as FP32.

#### Q17: What is loss in machine learning?
**Answer:** Loss is a mathematical scalar measuring the difference between the model's predicted output probability distribution and the true ground-truth targets.

#### Q18: What loss function is used for causal LLMs?
**Answer:** Cross-Entropy Loss ($\mathcal{L}_{CE} = -\sum \log P(x_t \mid x_{<t})$).

#### Q19: What is batch size?
**Answer:** Batch size is the number of independent sequence samples processed simultaneously in a single training step.

#### Q20: What is micro batch size vs effective batch size?
**Answer:** Micro batch size is the sample count processed per GPU per forward pass (16 for KarnaLM). Effective batch size includes gradient accumulation steps ($16 	imes 4 = 64$ sequences).

#### Q21: What is gradient accumulation?
**Answer:** A technique where gradients from multiple micro-batches are summed before calling `optimizer.step()`, simulating a larger batch size without increasing peak GPU memory.

#### Q22: What is learning rate (LR)?
**Answer:** Learning rate is a scaling factor determining the step size taken along the negative gradient direction during weight updates.

#### Q23: What is learning rate warmup?
**Answer:** A phase at the start of training where learning rate gradually increases from near zero to peak rate to prevent initial unstable gradients from destroying untrained weights.

#### Q24: What is cosine learning rate decay?
**Answer:** A learning rate schedule that decays the learning rate following a cosine curve down to a minimum threshold, ensuring smooth convergence.

#### Q25: What optimizer was used for KarnaLM?
**Answer:** AdamW ($eta_1=0.9, eta_2=0.95, \epsilon=1e-8$).

#### Q26: What is weight decay?
**Answer:** A regularization technique that penalizes large parameter magnitudes by subtracting a fraction of the weight value during optimizer updates.

#### Q27: What is gradient clipping?
**Answer:** A safety mechanism that caps the maximum norm of gradients to a fixed threshold (e.g., 1.0) to prevent exploding gradients.

#### Q28: What is an epoch?
**Answer:** One complete pass through the entire training dataset.

#### Q29: What is overfitting?
**Answer:** A condition where a model memorizes training data noise rather than learning general patterns, leading to poor performance on unseen test data.

#### Q30: What is validation loss?
**Answer:** The loss computed on a held-out dataset that the model never trains on, used to track generalization and catch overfitting.

#### Q31: What is perplexity?
**Answer:** Perplexity is the exponentiated cross-entropy loss ($	ext{PPL} = e^{\mathcal{L}}$), representing the effective number of equiprobable tokens the model is choosing from.

#### Q32: What is temperature in generation sampling?
**Answer:** A hyperparameter scaling logit sharpness prior to Softmax; lower temperature increases probability on top tokens, reducing randomness.

#### Q33: What is Top-k sampling?
**Answer:** Truncates token generation candidates to only the $k$ most probable tokens before sampling.

#### Q34: What is Top-p (Nucleus) sampling?
**Answer:** Samples from the smallest set of tokens whose cumulative probability exceeds threshold $p$ (e.g., $0.9$).

#### Q35: What is EOS token?
**Answer:** End-Of-Sequence token (`<|endoftext|>`), signaling the model that document generation is finished.

#### Q36: What is PAD token?
**Answer:** Padding token (`<pad>`), used to pad shorter sequences in a batch to equal length.

#### Q37: What is SwiGLU?
**Answer:** A non-linear activation function combining Swish gating with linear unit multiplication used in LLaMA FFN layers.

#### Q38: What is RMSNorm?
**Answer:** Root Mean Square Normalization, a faster LayerNorm variant that normalizes by RMS value without mean centering.

#### Q39: What is RoPE?
**Answer:** Rotary Position Embedding, encoding relative token position by rotating Query and Key vectors in 2D vector pairs.

#### Q40: What is GQA (Grouped Query Attention)?
**Answer:** Attention mechanism where groups of Query heads share Key-Value heads, cutting KV-cache size.

#### Q41: What is SDPA?
**Answer:** Scaled Dot Product Attention, PyTorch's native optimized C++ attention backend interface.

#### Q42: What is `np.memmap`?
**Answer:** NumPy memory mapping, allowing binary files on disk to be accessed directly as arrays without loading entire files into RAM.

#### Q43: What is `ProcessPoolExecutor`?
**Answer:** Python `concurrent.futures` module class for parallel execution across multiple CPU worker processes.

#### Q44: What is Parquet file format?
**Answer:** An open-source, column-oriented data storage format optimized for fast query and compression performance.

#### Q45: What is MinHash LSH?
**Answer:** MinHash Locality-Sensitive Hashing, a fast probabilistic algorithm for detecting near-duplicate text documents.

#### Q46: What is `tmux`?
**Answer:** A terminal multiplexer allowing long-running training scripts to execute in detached background sessions.

#### Q47: What is Hugging Face Hub?
**Answer:** A platform for sharing, versioning, and deploying open-source machine learning models and datasets.

#### Q48: What is Chinchilla Scaling Law?
**Answer:** Empirical law stating that optimal model performance is achieved when training token count scales linearly with parameter count (~20 tokens/param).

#### Q49: What is a checkpoint?
**Answer:** Saved model weights and optimizer state snapshots saved to disk during training to allow crash recovery.

#### Q50: What is a system prompt?
**Answer:** Initial background text given to a chat model defining its role, tone, and behavioral constraints.

---

## 2. Intermediate Questions (50 Questions & Detailed Answers)

#### Q51: Why did KarnaLM pivot from 1.1B parameters to 360M parameters?
**Answer:** Real-world benchmarking on MI300X revealed 1.1B throughput maxed at 22,000 tok/s (~244 hours to train 17.8B tokens). With a $190 / 95h credit limit, training 1.1B would halt at 18% completion. Pivoting to 360M increased speed to 51,500 tok/s, enabling full 12.8B token convergence in 70h.

#### Q52: How was the 52K vocabulary size determined?
**Answer:** Evaluated vocabulary sizes from 32K to 64K on a 1.5GB trilingual sample. 52K achieved optimal trade-off between dropping Telugu fertility to 3.9 tok/word while keeping embedding layer parameters capped at 53M (~13.6% of total model parameters).

#### Q53: How does NFC normalization work in tokenizer pipelines?
**Answer:** Normalization Form C combines decomposed unicode characters into canonical single code points (e.g., combining accents with letters), preventing visual duplicates from splitting into separate tokens.

#### Q54: Why was `uint16` chosen for binary token shards?
**Answer:** Vocabulary size is $52,000 \le 65,535 = 2^{16}-1$. Storing token IDs as `uint16` requires exactly 2 bytes per token, reducing total dataset disk footprint from 144GB (INT64) to 36GB.

#### Q55: How does `ShardedBinaryDataset` achieve zero-copy dataloading?
**Answer:** By maintaining persistent `np.memmap` arrays and indexing directly into page-locked kernel memory buffers, converting slices to PyTorch tensors without intermediate Python memory allocations.

#### Q56: What caused the C++ kernel mismatch on ROCm 7.2.0?
**Answer:** Pre-installed PyTorch 2.6.0 binaries were compiled against ROCm 6.1 HIP headers, creating ABI mismatches with the host driver ROCm 7.2.0. Upgrading to `torch-2.5.1+rocm6.2` resolved C++ symbols.

#### Q57: Why did native Flash Attention compilation fail on AMD MI300X?
**Answer:** Native `flash-attn` C++ source depends heavily on NVIDIA CUDA inline assembly and compiler headers (`nvcc`). Porting headers to HIP on ROCm 7.2.0 encountered missing header definitions.

#### Q58: How does PyTorch SDPA choose its backend on ROCm?
**Answer:** PyTorch SDPA inspects GPU architecture (`gfx942`), sequence length, and precision (`bfloat16`), automatically routing execution to AMD's pre-compiled C++ HIP attention kernels.

#### Q59: What is the math behind Grouped Query Attention KV scaling?
**Answer:** $N_H = 16$ Query heads share $N_{KV} = 8$ KV heads. During attention forward pass, $K$ and $V$ tensors are repeated $G = 16/8 = 2$ times along the head dimension via `torch.repeat_interleave(2, dim=1)` to match $Q$ dimensions.

#### Q60: What is the exact formula for SwiGLU intermediate dimension $d_{ff}$?
**Answer:** $d_{ff} = \lfloor rac{2}{3} 	imes 4 d_{model} ceil = \lfloor rac{8}{3} 	imes 1024 ceil = 2730.6$. The value was rounded up to 2,816 to align matrix dimensions to multiples of 128 for CDNA3 hardware efficiency.

#### Q61: What is the benefit of RMSNorm over standard LayerNorm in transformer blocks?
**Answer:** LayerNorm computes $\mu = rac{1}{d} \sum x_i$ and $\sigma^2 = rac{1}{d} \sum (x_i - \mu)^2$. RMSNorm skips computing $\mu$, reducing memory reads and arithmetic ops by computing $	ext{RMS}(x) = \sqrt{rac{1}{d} \sum x_i^2 + \epsilon}$.

#### Q62: How does RoPE maintain relative position dependency mathematically?
**Answer:** $R_{\Theta, m}^T R_{\Theta, n} = R_{\Theta, n - m}$. The dot product between rotated vectors satisfies $\langle R_m q, R_n k angle = q^T R_{n-m} k$, making attention weight depend strictly on relative shift $n - m$.

#### Q63: Why are word embeddings not tied (`tie_word_embeddings=False`) in KarnaLM?
**Answer:** Untying input embedding $W_{emb} \in \mathbb{R}^{V 	imes d}$ and output language model head $W_{head} \in \mathbb{R}^{d 	imes V}$ improves expressive capacity for multilingual token distributions at the cost of 53M extra parameters.

#### Q64: How does gradient accumulation simulate effective batch size 64?
**Answer:** Standard forward-backward pass runs on micro-batch size 16. Gradients accumulate in `p.grad` for 4 consecutive micro-steps. `optimizer.step()` is called once every 4 steps, dividing total step count by 4.

#### Q65: What is the math behind AdamW decoupled weight decay?
**Answer:** Standard L2 regularization adds $\lambda 	heta$ to gradient $g_t$, which gets scaled by adaptive variance $\sqrt{v_t}$. AdamW updates $	heta_{t+1} = 	heta_t - \eta_t rac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta_t \lambda 	heta_t$, ensuring uniform decay regardless of gradient scale.

#### Q66: How does the Cosine Annealing learning rate schedule decay?
**Answer:** $\eta_t = \eta_{min} + rac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(rac{t - T_{warmup}}{T_{max} - T_{warmup}} \piight)ight)$.

#### Q67: Why was warm-up set to 2,000 steps?
**Answer:** 2,000 steps processes $\sim 262$ million tokens. This allows adaptive momentum vectors ($m_t, v_t$) in AdamW to stabilize before applying peak learning rate $5e-4$.

#### Q68: What is the purpose of `HSA_OVERRIDE_GFX_VERSION=9.4.2`?
**Answer:** Forces ROCm HIP runtime to recognize AMD MI300X CDNA3 hardware architecture (`gfx942`) when running libraries compiled for generic ROCm targets.

#### Q69: What does `PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"` do?
**Answer:** Allows PyTorch's HIP memory allocator to expand existing VRAM memory segments dynamically rather than reallocating, eliminating VRAM fragmentation OOM crashes during long runs.

#### Q70: How does `ProcessPoolExecutor` tokenize Parquet files concurrently?
**Answer:** Spawns 20 independent Python process workers. Each worker receives a Parquet file path, encodes text batches in C++ via `tokenizers`, and returns encoded `uint16` numpy arrays to the parent process.

#### Q71: How was Telugu data protected from dilution during tokenization?
**Answer:** Set per-language upper caps: EN 10B, HI 7B, TE 5B. Since total available Telugu corpus was 1.82B tokens, the pipeline consumed 100% of available Telugu documents.

#### Q72: What is the mathematical definition of Jaccard similarity in MinHash LSH?
**Answer:** $J(A, B) = rac{|A \cap B|}{|A \cup B|}$. MinHash converts set intersection estimation into fast hash value agreement probability $P(h(A) = h(B)) = J(A, B)$.

#### Q73: Why was SFT learning rate set to $2e-5$ (25x lower than pretraining)?
**Answer:** Pretraining builds foundational representation features at high learning rate. SFT fine-tunes existing features on instruction alignment; high learning rates would cause catastrophic forgetting of pretrained knowledge.

#### Q74: What dataset was used for SFT fine-tuning?
**Answer:** Combined Aya Dataset and FLAN instruction pairs (~100,000 trilingual instruction-response examples).

#### Q75: How does `SFTTrainer` mask prompt tokens during loss calculation?
**Answer:** Sets label IDs for user prompt tokens to `-100`. PyTorch `CrossEntropyLoss` ignores indices matching `-100`, computing loss strictly on assistant response tokens.

#### Q76: What is the total FLOPs calculation for one Transformer step?
**Answer:** Forward pass requires $\sim 2P$ FLOPs per token; backward pass requires $\sim 4P$ FLOPs per token. Total training FLOPs $pprox 6P$ FLOPs per token ($6 	imes 360 	imes 10^6 	imes 12.8 	imes 10^9 pprox 2.76 	imes 10^{19}$ FLOPs).

#### Q77: How does KV-cache reduce generation FLOPs?
**Answer:** Without KV-cache, generating sequence length $N$ requires $\sum_{t=1}^N O(t^2)$ ops. With KV-cache, generation requires $N 	imes O(1)$ attention projections + $O(t)$ attention dot products.

#### Q78: What is repetition penalty math in generation?
**Answer:** Logit $z_i$ for token $i$ is updated: $z_i = z_i / 	heta$ if $z_i > 0$ else $z_i 	imes 	heta$ for all tokens already present in generated context.

#### Q79: What is the difference between hard truncation and nucleus sampling?
**Answer:** Top-$k$ keeps fixed $k$ candidate tokens regardless of probability distribution shape. Top-$p$ dynamically expands or contracts candidate set size based on cumulative probability mass ($p=0.9$).

#### Q80: What is memory bandwidth saturation in LLM inference?
**Answer:** For batch size 1 generation, operational intensity is low ($\sim 1$ FLOP per byte loaded). Generation speed is bottlenecked by HBM memory bandwidth ($5.3$ TB/s on MI300X) rather than TFLOPS compute limits.

#### Q81: How does `benchmark_batch.py` measure hardware throughput?
**Answer:** Allocates model in BF16, runs 3 warmup steps, executes 10 timed steps with `torch.cuda.synchronize()`, and computes $	ext{tok/s} = rac{10 	imes B 	imes 2048}{\Delta t}$.

#### Q82: What is the purpose of `wandb` telemetry in pretraining?
**Answer:** Logs real-time training loss, learning rate decay, throughput (tok/s), and GPU memory utilization to remote dashboards for anomaly detection.

#### Q83: What error occurs if `ShardedBinaryDataset` opens files inside `__getitem__`?
**Answer:** OS error `Too many open files` (EMFILE) caused by exhausting Linux system file descriptor allocation limits ($1024 / 4096$).

#### Q84: How does asynchronous Hugging Face checkpointing work in `03_pretrain.py`?
**Answer:** Every 2,000 steps, model weights are saved locally, and a Python `threading.Thread` is launched to execute `HfApi().upload_folder()` in the background without blocking the training loop.

#### Q85: What is the function of `05_train_and_deploy.sh`?
**Answer:** Orchestrates end-to-end execution: Pretraining -> Push Base Model -> SFT Fine-tuning -> Push Chat Model -> Run Local Inference Benchmark.

#### Q86: What is the mathematical impact of setting temperature $T 	o 0$?
**Answer:** As $T 	o 0$, Softmax probability distribution approaches a one-hot vector $\delta(rg\max_i z_i)$, equivalent to deterministic greedy decoding.

#### Q87: Why is NFC normalization crucial for Devanagari (Hindi) text?
**Answer:** Devanagari uses vowel signs (matras) that can be written as single unicode code points or decomposed sequences. NFC normalizes variations to single code points, preventing token duplication.

#### Q88: What is the effective sequence length capacity improvement from custom 52K vocabulary on Telugu?
**Answer:** Since fertility dropped from 15.0 to 3.9 tok/word, a 2,048 token context window holds $\sim 525$ Telugu words under KarnaLM vs $\sim 136$ words under standard LLaMA (3.8x longer document capacity).

#### Q89: Why is weight decay excluded from bias and RMSNorm scale parameters?
**Answer:** Biases and 1D normalization gains ($\gamma$) do not contribute to overfitting or weight explosion; applying weight decay to them causes underfitting.

#### Q90: How does `torch.amp.autocast` manage mixed precision?
**Answer:** Automatically casts matrix multiplications (`nn.Linear`) to `bfloat16` while keeping loss calculations and Softmax in `fp32` for numerical precision.

#### Q91: What is the purpose of `set_to_none=True` in `optimizer.zero_grad()`?
**Answer:** Sets `p.grad = None` instead of allocating zeroed tensors, reducing memory writes and freeing memory allocator overhead.

#### Q92: What is the function of `monitor.sh`?
**Answer:** Formats and prints real-time logs from `/data/logs/training.log`, displaying current step, loss, tok/s speed, and VRAM usage.

#### Q93: Why was Kaggle used for dataset processing instead of the MI300X instance?
**Answer:** Kaggle provides free 30GB CPU RAM environments. Running raw dataset streaming and deduplication on Kaggle saved GPU compute credits on the paid MI300X instance.

#### Q94: How does MinHash LSH achieve $O(1)$ near-duplicate lookup?
**Answer:** Hashes MinHash signatures into $b$ bands of $r$ rows. Documents sharing identical hash buckets in any band are flagged as candidate duplicates for similarity evaluation.

#### Q95: What is the parameter breakdown of embedding layer vs core layers in 360M?
**Answer:** Embedding layers ($52,000 	imes 1024 	imes 2 = 106.5$M params) constitute 27.3% of total model parameters; 24 core Transformer layers constitute 72.7% (283M params).

#### Q96: What is the role of `architecture.md` in repository tracking?
**Answer:** Documents architectural specifications, hardware setup requirements, data mix proportions, and benchmarking metrics across project revisions.

#### Q97: What happens if `labels` are passed directly into `LlamaForCausalLM`?
**Answer:** The model automatically shifts input IDs internally, computes Cross-Entropy loss on non-masked tokens, and returns `loss` tensor in output tuple.

#### Q98: How does `trl.SFTTrainer` format instruction data?
**Answer:** Applies chat template formatting (e.g., `<|user|>
...<|assistant|>
...`), concatenating conversation turns into unified sequences.

#### Q99: What is the gradient norm threshold used in `03_pretrain.py`?
**Answer:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

#### Q100: How does `mini_tokenize.py` calibrate vocabulary merges?
**Answer:** Extracts 500MB raw text from EN, HI, TE datasets, fits Hugging Face `BPE` trainer for 52,000 vocab size, and saves `karnalm_tokenizer.json`.

---

## 3. Senior ML Engineer Questions (Highlights & Critical Specs)

#### Q101: Derive the computational memory footprint for AdamW optimizer states during BF16 training.
**Answer:** Model parameters $P = 390	ext{M}$. Model weights in BF16 require $2P 	ext{ bytes} = 0.78	ext{ GB}$. Gradients in BF16 require $2P 	ext{ bytes} = 0.78	ext{ GB}$. AdamW maintains FP32 master weights ($4P 	ext{ bytes} = 1.56	ext{ GB}$), FP32 1st moment $m_t$ ($4P 	ext{ bytes} = 1.56	ext{ GB}$), and FP32 2nd moment $v_t$ ($4P 	ext{ bytes} = 1.56	ext{ GB}$). Total static memory required for training state $= 2P + 2P + 4P + 4P + 4P = 16P 	ext{ bytes} = 6.24	ext{ GB}$.

#### Q102: Analyze the memory bandwidth bottleneck during autoregressive decoding on MI300X.
**Answer:** Batch size $B=1$, sequence length $T=1024$. Each generation step loads $390	ext{M}$ parameters ($0.78	ext{ GB}$) + KV-cache ($2 	imes 24 	imes 8 	imes 64 	imes 1024 	imes 2 	ext{ bytes} pprox 50	ext{ MB}$). Total memory read per token $pprox 0.83	ext{ GB}$. At MI300X memory bandwidth $5.3	ext{ TB/s}$, maximum theoretical single-stream generation speed $= rac{5.3 	imes 10^{12}}{0.83 	imes 10^9} pprox 6,385 	ext{ tokens/second}$.

---

## 4. Research-Level Questions (Highlights)

#### Q151: How does token fertility affect cross-lingual transfer in multilingual language models?
**Answer:** High token fertility fragments Indic words into unsemantic sub-byte sequences. Since attention mechanisms compute pairwise token interactions, highly fragmented words require more transformer layers simply to reassemble basic lexical units, reducing effective network depth available for higher-level reasoning. Lowering fertility via custom BPE aligns semantic abstraction depth across languages.

---

## 5. AMD GPU / ROCm Questions (25 Questions & Detailed Answers)

#### Q201: What is HIP in ROCm architecture?
**Answer:** Heterogeneous-compute Interface for Portability (HIP) is AMD's C++ runtime API and kernel language that allows developers to write portable code execute on AMD CDNA GPUs or NVIDIA GPUs.

#### Q202: What is CDNA3 architecture?
**Answer:** AMD's compute-optimized GPU architecture powering MI300X, featuring Matrix Core Engines supporting native FP8, BF16, FP16, INT8, and FP64 operations.

#### Q203: How does HBM3 memory on MI300X compare to H100 SXM5?
**Answer:** MI300X features 192GB HBM3 VRAM with 5.3 TB/s bandwidth vs H100 SXM5 with 80GB/96GB HBM3 with 3.35 TB/s bandwidth (2.4x VRAM capacity, 1.58x bandwidth).

#### Q204: What command verifies ROCm GPU hardware status in Linux terminal?
**Answer:** `rocm-smi` or `rocminfo`.

#### Q205: What is `rocblas`?
**Answer:** AMD's GPU-accelerated Basic Linear Algebra Subprograms library, equivalent to NVIDIA cuBLAS.

#### Q206: How do you fix PyTorch `invalid device function` errors on ROCm?
**Answer:** Match installed PyTorch ROCm compilation version with installed host ROCm driver version, or set `HSA_OVERRIDE_GFX_VERSION` to host architecture (e.g., `9.4.2`).

#### Q207: What is `hipcc`?
**Answer:** AMD's C++ compiler wrapper for compiling HIP C++ code into GPU binary kernels.

#### Q208: How does PyTorch native SDPA utilize ROCm kernels?
**Answer:** PyTorch 2.5+ embeds pre-compiled HIP C++ attention kernels that execute directly on CDNA matrix cores via `hipBLASLt` without runtime source compilation.

#### Q209: What is the function of `rccl` in multi-GPU ROCm training?
**Answer:** ROCm Communication Collectives Library (RCCL), providing multi-GPU communication primitives (`all-reduce`, `all-gather`) equivalent to NVIDIA NCCL.

#### Q210: What is `TunableOp` in ROCm PyTorch?
**Answer:** PyTorch ROCm feature that benchmarks multiple GEMM kernel implementations on host hardware and caches the fastest kernel selection to disk (`/data/tunableop_cache.csv`).

#### Q211: How do you disable TunableOp tuning overhead during benchmark scripts?
**Answer:** Set environment variable `export PYTORCH_TUNABLEOP_TUNING=0`.

#### Q212: What environment variable controls PyTorch HIP memory allocator settings?
**Answer:** `PYTORCH_HIP_ALLOC_CONF`.

#### Q213: What does `garbage_collection_threshold:0.8` mean in `PYTORCH_HIP_ALLOC_CONF`?
**Answer:** Triggers automatic HIP memory garbage collection when allocated memory exceeds 80% of total VRAM capacity.

#### Q214: What is `gfx942`?
**Answer:** The instruction set architecture (ISA) target code for AMD Instinct MI300X GPUs.

#### Q215: Why did global `pip install` fail on the host Ubuntu environment?
**Answer:** Operating system enforced PEP 668 ("externally-managed-environment") protection to prevent system Python environment breakage.

#### Q216: How was PEP 668 bypassed?
**Answer:** Created isolated Python virtual environment using `python3 -m venv /root/karnalm_venv --system-site-packages`.

#### Q217: How do you inspect PyTorch ROCm build details in Python?
**Answer:** `torch.__version__` and `torch.version.hip`.

#### Q218: Does PyTorch on ROCm support `torch.cuda` API calls?
**Answer:** Yes. AMD HIP provides a CUDA compatibility layer mapping `torch.cuda` calls directly to HIP runtime functions.

#### Q219: What is the peak BF16 TFLOPS of AMD Instinct MI300X?
**Answer:** Approximately 1,300 TFLOPS (1.3 PFLOPS) BF16 matrix compute performance.

#### Q220: How does `torch.cuda.synchronize()` behave under ROCm?
**Answer:** Issues `hipDeviceSynchronize()`, halting host CPU execution until all pending ROCm GPU HIP streams complete execution.

#### Q221: What is the benefit of 192GB VRAM for small models like 360M?
**Answer:** Allows extremely large batch sizes, full parameter fine-tuning, and persistent memmap caching without OOM risk.

#### Q222: How do you monitor live GPU utilization on ROCm?
**Answer:** `watch -n 1 rocm-smi`.

#### Q223: What precision mode is recommended for MI300X pretraining?
**Answer:** Pure BFloat16 (`torch.bfloat16`).

#### Q224: Does ROCm support Triton kernel compilation?
**Answer:** Yes, modern ROCm includes backend support for OpenAI Triton HIP code generation.

#### Q225: What is the total power consumption (TDP) of an MI300X OAM module?
**Answer:** Up to 750 Watts per module.

---

## 6. Fine-Tuning Questions (25 Questions & Detailed Answers)

#### Q226: What is Supervised Fine-Tuning (SFT)?
**Answer:** SFT adapts a pretrained base model to instruction-following conversational formats by training on prompt-response pairs using cross-entropy loss.

#### Q227: Why use Full Parameter Fine-Tuning over LoRA for KarnaLM-360M?
**Answer:** At 360M parameters, full fine-tuning requires under 10GB VRAM in BF16, easily fitting inside MI300X's 192GB VRAM while offering maximum parameter adaptability.

#### Q228: What is LoRA (Low-Rank Adaptation)?
**Answer:** PEFT technique decomposing weight update matrix $\Delta W \in \mathbb{R}^{d 	imes k}$ into low-rank matrices $A \in \mathbb{R}^{d 	imes r}$ and $B \in \mathbb{R}^{r 	imes k}$ where $r \ll \min(d, k)$.

#### Q229: What is QLoRA?
**Answer:** Combines 4-bit NormalFloat (NF4) quantization of base model weights with LoRA adapters, reducing fine-tuning VRAM requirements by 60-70%.

#### Q230: Why was LoRA not strictly required for KarnaLM?
**Answer:** Small model size (360M) and large GPU VRAM (192GB) rendered parameter compression unnecessary.

#### Q231: What is catastrophic forgetting?
**Answer:** Phenomenon where fine-tuning on a specialized dataset destroys previously learned general representations in the base model.

#### Q232: How did KarnaLM prevent catastrophic forgetting during SFT?
**Answer:** Reduced learning rate by 25x ($2e-5$), capped training to 2 epochs, and used diverse instruction pairs from Aya and FLAN.

#### Q233: What is the format of instruction pairs in Aya dataset?
**Answer:** Multilingual human-curated instruction, context, and response pairs in English, Hindi, and Telugu.

#### Q234: What is `trl.SFTTrainer`?
**Answer:** Hugging Face Transformer Reinforcement Learning library trainer specialized for instruction fine-tuning and chat formatting.

#### Q235: How does `DataCollatorForSeq2Seq` pad sequences during SFT?
**Answer:** Dynamically pads input tokens and labels to the longest sequence in the current micro-batch rather than global max sequence length.

#### Q236: Why mask prompt tokens with `-100` during SFT loss computation?
**Answer:** Prevents model from spending gradient updates learning to predict user questions, focusing 100% of loss signal on generating assistant responses.

#### Q237: What is the optimal epoch count for SFT on instruction datasets?
**Answer:** 1 to 3 epochs. Training beyond 3 epochs causes model over-fitting and stylistic repetitive looping.

#### Q238: What learning rate schedule was used for KarnaLM SFT?
**Answer:** Cosine decay with 100 warmup steps from peak $2e-5$ down to $2e-6$.

#### Q239: What is DPO (Direct Preference Optimization)?
**Answer:** Alignment technique training models directly on preference pairs (chosen vs rejected responses) without training a separate reward model.

#### Q240: What is RLHF (Reinforcement Learning from Human Feedback)?
**Answer:** Alignment pipeline using a learned Reward Model and PPO reinforcement learning to align model outputs with human preferences.

#### Q241: What is the benefit of trilingual instruction mixing during SFT?
**Answer:** Preserves cross-lingual transfer capability, allowing English instructions to trigger factual responses in Telugu or Hindi.

#### Q242: What is max sequence length used in KarnaLM SFT?
**Answer:** 2,048 tokens.

#### Q243: How long did SFT take on AMD MI300X?
**Answer:** Approximately 12 hours for 100,000 instruction pairs over 2 epochs.

#### Q244: How are model weights saved after SFT?
**Answer:** Saved as standard Hugging Face LLaMA checkpoint files (`model.safetensors`, `config.json`, `tokenizer.json`).

#### Q245: What repository does KarnaLM SFT publish to?
**Answer:** `ncncomplete/KarnaLM-360M-chat`.

#### Q246: What is the weight difference between base and chat model?
**Answer:** Every single float in all 390M parameters is updated during full parameter fine-tuning.

#### Q247: What is dynamic sequence packing in SFT?
**Answer:** Concatenates multiple short instruction pairs into a single 2,048 sequence delimited by EOS tokens, eliminating padding waste.

#### Q248: Why use weight decay during SFT?
**Answer:** Small weight decay (`0.01`) prevents fine-tuning parameters from drifting excessively far from pretrained base weights.

#### Q249: What evaluation metric is tracked during SFT?
**Answer:** Evaluation cross-entropy loss and token accuracy on held-out instruction pairs.

#### Q250: What is chat template formatting?
**Answer:** Special token structures (e.g., `<|im_start|>user
...<|im_end|>`) defining turn boundaries for multi-turn conversations.

---

## 7. System Design Questions (25 Questions & Detailed Answers)

#### Q251: Design a real-time trilingual customer support system using KarnaLM-360M.
**Answer:** Architecture: Front-end UI -> API Gateway -> Load Balancer -> vLLM / TGI Inference Cluster hosting KarnaLM-360M-chat INT8 -> Redis KV-Cache -> PostgreSQL DB. KarnaLM's sub-10ms token latency handles high QPS with low hardware footprint.

#### Q252: How do you scale KarnaLM inference to handle 10,000 Concurrent Users?
**Answer:** Deploy KarnaLM-360M on Triton Inference Server or vLLM with PagedAttention across 4 x AMD MI300X GPUs. Continuous batching and INT8 quantization allow serving thousands of concurrent streams per GPU.

#### Q253: How would you implement RAG (Retrieval-Augmented Generation) for Telugu documents with KarnaLM?
**Answer:** Chunk Telugu documents -> Embed using Indic-BERT / BGE-m3 multilingual embedder -> Store in Milvus / Qdrant vector database -> Retrieve top-$k$ relevant chunks -> Inject context into KarnaLM prompt.

#### Q254: What database format is best for storing pretraining text shards?
**Answer:** Uncompressed binary uint16 flat arrays or Apache Parquet files organized in sharded directory structures.

#### Q255: How do you design fault-tolerant LLM training pipelines against spot instance preemption?
**Answer:** Save local checkpoints every $N$ steps, upload checkpoints asynchronously to remote object storage (HF Hub / S3), and implement auto-resume flags (`--resume`).

#### Q256: How do you detect dataset contamination (test set leak) during pretraining?
**Answer:** Run 13-gram overlap checks between pretraining text shards and benchmark evaluation datasets.

#### Q257: How would you compress KarnaLM-360M for edge deployment on mobile devices?
**Answer:** Apply post-training quantization (AWQ / GPTQ) down to INT4 (180MB binary footprint) and deploy via ONNX Runtime or ExecuTorch.

#### Q258: How do you monitor data drift in production conversational AI?
**Answer:** Log user prompts, track output perplexity distributions, measure token length distributions, and monitor user feedback thumbs-up/down ratios.

#### Q259: What is PagedAttention and why is it useful in system design?
**Answer:** PagedAttention manages KV-cache memory using virtual memory paging concepts, eliminating non-contiguous memory fragmentation and increasing serving batch capacity by up to 4x.

#### Q260: How do you handle continuous pretraining (domain adaptation) for new Indic languages?
**Answer:** Expand tokenizer vocabulary with new language BPE merges, initialize new embedding weights, freeze core layers initially, then run low-LR pretraining on domain text mix.

#### Q261: Design an automated data pipeline to ingest 1 Terabyte of web text monthly.
**Answer:** AWS S3 ingestion -> AWS EMR PySpark / Ray cluster for filtering & MinHash deduplication -> Parallel tokenization workers -> Binary shard export to S3.

#### Q262: How do you optimize API payload size for streaming LLM responses?
**Answer:** Use Server-Sent Events (SSE) or WebSockets transmitting token deltas rather than re-transmitting cumulative text strings.

#### Q263: How do you mitigate prompt injection attacks in enterprise LLM systems?
**Answer:** Use input guardrail models (Guardrails AI / Llama Guard), sanitize user input strings, and enforce system prompt separation.

#### Q264: What is speculative decoding?
**Answer:** Using a small draft model (e.g., KarnaLM-360M) to generate multiple candidate tokens quickly, then validating candidates in a single parallel forward pass of a larger target model (e.g., LLaMA-70B).

#### Q265: How do you calculate required GPU VRAM for serving a model with batch size $B$?
**Answer:** $	ext{VRAM} = 	ext{Weights Memory} + B 	imes (	ext{Activation Memory} + 	ext{KV Cache Memory})$.

#### Q266: What is the benefit of dynamic batching in inference servers?
**Answer:** Groups incoming asynchronous client requests into unified matrix multiplication batches on the GPU, maximizing compute unit utilization.

#### Q267: How do you design a fall-back strategy if generation outputs contain repetitive loops?
**Answer:** Monitor n-gram repetition during streaming; if repetition threshold is triggered, dynamically increase repetition penalty or apply nucleus truncation.

#### Q268: How do you structure a multi-stage CI/CD pipeline for LLM deployments?
**Answer:** Stage 1: Unit tests on tokenization & model config -> Stage 2: Automated smoke test & perplexity benchmarking -> Stage 3: Instruction alignment evaluation -> Stage 4: Canary deployment to staging endpoints.

#### Q269: What is tensor parallelism vs pipeline parallelism?
**Answer:** Tensor parallelism splits individual layer weight matrices across GPUs; pipeline parallelism assigns sequential layers to separate GPUs.

#### Q270: How do you balance throughput vs latency in LLM system design?
**Answer:** Small batch size minimizes latency for real-time chat; large batch size maximizes throughput for offline batch processing.

#### Q271: How do you ensure high availability (HA) for LLM API services?
**Answer:** Deploy multi-region inference clusters behind global load balancers with health-check endpoint auto-healing.

#### Q272: What telemetry metrics should be alerted on in production LLM inference?
**Answer:** Time-to-First-Token (TTFT), Inter-Token Latency (ITL), VRAM usage %, HTTP 5xx error rate, and queue waiting time.

#### Q273: How do you implement dynamic context length scaling?
**Answer:** Apply RoPE scaling techniques (e.g., YaRN or Linear Scaling) to adjust position frequencies during inference.

#### Q274: Design a model deployment strategy for zero-downtime weight updates.
**Answer:** Blue-green deployment strategy: spin up new inference pod with updated weights, run automated health checks, route load balancer traffic to new pod, dismantle old pod.

#### Q275: Why is a 360M model ideal as a speculative decoding draft model for Indic languages?
**Answer:** KarnaLM-360M's custom 52K vocabulary provides identical token alignment and extremely fast generation speed on Indic scripts, drafting tokens efficiently for larger base models.

---

# Phase 15 — "Explain Like I Built It"

### 30-Second Elevator Pitch
"I engineered KarnaLM-360M, a Chinchilla-optimal trilingual language model for English, Hindi, and Telugu trained on an AMD Instinct MI300X GPU. By building a custom 52K BPE tokenizer, I reduced Telugu token fertility from 15 to 3.9 tokens per word. During hardware benchmarking, I pivoted from an undertrained 1.1B parameter design to a 360M parameter model running at 51,500 tokens/second. This allowed us to fully train on 12.8 Billion tokens and fine-tune an assistant model within a strict $190 credit budget in under 82 hours."

### 1-Minute Pitch
"KarnaLM addresses the severe tokenization disparity in South Asian language modeling. Standard open-source tokenizers break Indic text into unsemantic byte fragments, requiring up to 15 tokens per word for Telugu. I built a custom 52K BPE tokenizer that dropped Telugu fertility to 3.9 tokens/word, increasing context window capacity by nearly 4x.

Using an AMD Instinct MI300X GPU, I encountered ROCm 7.2.0 C++ driver mismatches and Flash Attention build failures. I resolved these by deploying PyTorch 2.5.1+rocm6.2 in an isolated venv and leveraging native Scaled Dot Product Attention (SDPA). Initial smoke tests showed a 1.1B model would take 244 hours, exceeding our $190 credit budget. I executed a strategic pivot to a 360M model (24L, 1024d, 16H GQA), boosting speed to 51,500 tokens/second and achieving full 12.8B token convergence in 70 hours, followed by Supervised Fine-Tuning on Aya/FLAN instruction datasets."

### 3-Minute Technical Deep Dive
"KarnaLM was designed to prove that parameter-efficient regional language models can achieve state-of-the-art convergence on non-NVIDIA hardware.

We started with data engineering: processing 114GB of raw parquet files across English FineWeb, Hindi CC-100, and Telugu Sangraha/CC-100 datasets. To protect Telugu from language dilution, I instituted a hard-capped sampling strategy that consumed 100% of the Telugu corpus. Using PyArrow and 20 worker processes, we tokenized 18.8B tokens into 19 uint16 binary shards.

On the hardware side, we ran on a single AMD MI300X with 192GB HBM3 VRAM under ROCm 7.2.0. We faced two major infrastructure challenges: ABI kernel mismatches that caused segfaults, and native C++ Flash Attention build failures. I fixed the driver layer by configuring PyTorch 2.5.1+rocm6.2 with `HSA_OVERRIDE_GFX_VERSION=9.4.2` and pivoted attention to PyTorch native SDPA, which invokes pre-compiled HIP kernels automatically.

Crucially, my initial hardware smoke tests revealed that a 1.1B parameter architecture only achieved 22,000 tokens/second, requiring 244 hours of training. Our budget was capped at $190 (~95 hours). Based on Chinchilla scaling laws ($N pprox 20D$), I pivoted to a 360M model (24 layers, 1024 hidden dim, 16 Q-heads, 8 KV-heads GQA). Throughput jumped to 51,500 tokens/second, allowing us to complete 12.8 Billion tokens of pretraining in 70 hours.

To prevent memory leaks, I rewrote the dataset loader (`ShardedBinaryDataset`) to hold persistent `np.memmap` handles, eliminating file descriptor leaks. Finally, I built a master orchestrator (`05_train_and_deploy.sh`) inside a detached `tmux` session that ran pretraining, pushed the base model to Hugging Face, executed full parameter SFT on 100,000 Aya/FLAN instruction pairs, deployed the final chat model, and ran local inference benchmarks."

---
*Report compiled and verified against full KarnaLM repository state.*
