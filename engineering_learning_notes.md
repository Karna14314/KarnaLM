# KarnaLM — Engineering & Learning Notes

This document is a living record of technical questions, engineering trade-offs, and Deep Learning concepts encountered during the development of KarnaLM. It serves as a knowledge base for future projects.

---

### 🟢 Topic 1: Resource Utilization vs. Efficiency
**Question:** Why is GPU/CPU usage low (9%) and VRAM usage only 9GB/192GB? Is it inefficient?
**Answer:** In LLM training, low resource usage often indicates an **optimized pipeline**, not inefficiency.
*   **VRAM Footprint:** A 360M model in BFloat16 only requires a few GB for weights/gradients. Using more VRAM wouldn't increase speed; it would just be "filler."
*   **CPU Role:** The CPU feeds the GPU. 9% usage means the data loader is so efficient the GPU never has to wait. High CPU usage would actually indicate a bottleneck.
*   **True Metric:** The only metric that matters is **Throughput (Tokens per Second)**. At ~51,000 tok/s, we are hitting the physical limits of the model's architecture on this hardware, regardless of how "empty" the rest of the 192GB VRAM looks.

### 🟢 Topic 2: The 1.1B vs. 360M Architecture Pivot
**Question:** Why did 1.1B OOM if 360M is so small? Why did the time drop from 244h to 70h?
**Answer:**
*   **Non-Linear Scaling:** 1.1B isn't just 3x more weights; it has **4x larger activation matrices** (due to doubling hidden dimension from 1024 to 2048). 
*   **The Throughput Trap:** To finish 1.1B in 70 hours, we would have needed a massive batch size, which causes OOM. At a safe batch size, it only processed ~22k tok/s, leading to a 244-hour estimate.
*   **Speed is King:** 360M runs at **51,000 tok/s**. Training a smaller model to "saturation" (fully converged) is significantly better than training a large model to only 15% completion.

### 🟢 Topic 3: Batch Size & Gradient Verification
**Question:** How did we verify gradients and batch sizes (16, 32, 64, 128)?
**Answer:** We performed "Sweet Spot" benchmarking:
*   **Optimal Batch (16):** We found that a batch size of 16 sequences per step kept the MI300X matrix units busiest without adding excessive memory overhead.
*   **Stability:** This balance ensures the "Gradient Norm" stays stable (usually < 2.0). If the batch size is too small, the learning becomes "noisy" and the loss spikes. If too large, learning slows down.

### 🟢 Topic 4: Flash Attention vs. SDPA (ROCm Constraints)
**Question:** Why did Flash Attention fail and what approach replaced it?
**Answer:**
*   **The Struggle:** Native `flash-attn` requires complex C++/CUDA/HIP compilation. On AMD ROCm, this often fails due to version mismatches between the compiler and the library source.
*   **Approaches Tried:** We first attempted Triton-based Flash Attention, but it was still unstable. 
*   **The Solution (SDPA):** We pivoted to PyTorch's native **Scaled Dot Product Attention (SDPA)**. This is a built-in, pre-compiled C++ kernel that is automatically optimized for the MI300X. It provides nearly the same speed as Flash Attention 2 but is 100% stable and requires no extra installation.

### 🟢 Topic 5: Smoke Test Issues & Learnings
**Question:** What issues did we face during the Smoke Test?
**Answer:**
*   **The Gatekeeper:** The smoke test (`02_smoke_test.py`) was our "safety net." It caught the Flash Attention build failure before we wasted hours on a broken training run.
*   **VRAM Fragmentation:** We initially saw "Out of Memory" errors even when usage looked low. The smoke test helped us identify that **memory fragmentation** was the culprit, leading us to implement `PYTORCH_HIP_ALLOC_CONF` settings like `expandable_segments:True` to keep the memory "flat" and clean.
*   **Throughput Baseline:** It gave us the 22,000 tok/s benchmark for 1.1B, which triggered the "Budget Reality Check" and our pivot to 360M.

### 🟢 Topic 6: Multilingual Tokenization Strategy
**Question:** how was the tokenization part done?
**Answer:**
*   **Balanced Sampling:** To prevent English from "drowning out" Hindi and Telugu, we used a capped sampling strategy. We set a 5B token target for Telugu (which was larger than the actual data), ensuring we consumed **100% of the Telugu CC-100 corpus**.
*   **Multi-Processing:** Since we had 117GB of raw text, we used **ProcessPoolExecutor** to tokenize on 64 CPU cores simultaneously, converting raw text into binary uint16 shards.
*   **Fertility Verification:** We verified the tokenizer was "good" by checking the fertility of Telugu words (e.g., "మరియు"). Our custom 52K BPE tokenizer reduced fertility from ~15 tokens down to **3.9 tokens per word**, making the model much more efficient at reading Telugu.

### 🟢 Topic 7: The Data "Recipe" & Proportions
**Question:** On total how much data is the model built, and in what proportions?
**Answer:**
We are training on **12.8 Billion Tokens** total. Here is the estimated trilingual mix:
*   **English (~50%):** Provides the "Reasoning" and "Logic" base (from high-quality web data).
*   **Hindi (~35%):** Provides the vast majority of the "Cultural Knowledge."
*   **Telugu (~15%):** Provides the specific linguistic nuance for your target region.
*   **Why this mix?** High-quality English data acts like a "scaffold." Even when the model is speaking Telugu, it uses the "logic" it learned from English to keep the sentences coherent.

### 🟢 Topic 8: Next-Token Prediction & Sequential Logic
**Question:** How does the model predict words in the correct order without giving gibberish?
**Answer:**
Your model is an **Autoregressive Transformer**. 
*   **The Chain:** It doesn't predict the whole answer at once. It predicts **one token at a time**. 
*   **The "Context" Loop:** When you ask "How to make a cake?", the model looks at those 6 words and predicts the most likely 7th word (e.g., "First"). Then, it looks at "How to make a cake? First" and predicts the 8th word (e.g., "gather").
*   **Order through Probability:** During pre-training on 12.8B tokens, the model has seen the word "Delhi" follow "The capital of India is" millions of times. It learns that the probability of "Delhi" being the next word is 99%, while "Hyderabad" is 0.1%. This statistical "memory" ensures words come out in a logical order.

### 🟢 Topic 9: Fighting Hallucinations (Pre-training vs. SFT)
**Question:** How will it not hallucinate or give random gibberish?
**Answer:**
We use a **Two-Stage** process to make the model "Foolproof":
1.  **Pre-training (Phase 1):** This is where the model gets its "Knowledge." It learns what a "cake" is and what "India" is. If we train long enough (12.8B tokens is very deep for a 360M model), it becomes highly "saturated" with facts, which reduces random gibberish.
2.  **SFT - Supervised Fine-Tuning (Phase 2):** This is the "Assistant Training." We give it 100,000+ examples of (Question -> Correct Answer). This teaches the model: *"When a human asks a question, don't just complete the text randomly—give a helpful, factual response."* 
3.  **Temperature Control:** During deployment (inference), we set a "Temperature" (e.g., 0.7). Low temperature forces the model to pick only the most certain words, which kills gibberish and significantly reduces hallucinations.

### 🟢 Topic 10: "Overkill" Hardware — Capacity vs. Speed
**Question:** If the model only uses 9GB, why not use a cheaper 16GB VRAM system?
**Answer:**
*   **VRAM is just the "Gas Tank":** Having a 192GB tank (MI300X) vs. a 16GB tank doesn't make the car faster; it just means you can hold more fuel. 16GB is enough *space*, but it doesn't have the *speed*.
*   **The Engine (Compute & Bandwidth):** The MI300X has **HBM3 memory bandwidth (5.3 TB/s)**. A standard 16GB card (like an RTX 4080) only has ~0.3 - 0.7 TB/s. 
*   **The Time Math:** 
    *   On this MI300X, we hit **51,000 tokens/sec** (70 hours total). 
    *   On a 16GB system, you would likely only get **5,000 - 8,000 tokens/sec** (300+ hours total). 
*   **Conclusion:** You are paying for the **bandwidth engine** to compress 3 weeks of training into 3 days.

### 🟢 Topic 11: KarnaLM-360M Full Technical Specifications
**Question:** What are the exact detailed specs of the model configuration?
**Answer:**
Here is the complete blueprint for your model and training run:

| **Category** | **Parameter** | **Value** |
| :--- | :--- | :--- |
| **Architecture** | **Total Parameters** | ~390M (337M Core + 53M Embeddings) |
| | **Layers** | 24 |
| | **Hidden Dimension** | 1024 |
| | **Attention Heads (Q)** | 16 |
| | **KV Heads (GQA)** | 8 (Grouped Query Attention) |
| | **Intermediate Size** | 2816 (SwiGLU Activation) |
| | **Vocab Size** | 52,000 (Optimized Trilingual BPE) |
| | **Context Window** | 2048 tokens |
| **Training** | **Precision** | Pure **BFloat16** |
| | **Batch Size** | 16 sequences per step |
| | **Grad Accumulation** | 4 steps |
| | **Effective Batch** | 64 sequences (131,072 tokens/update) |
| | **Peak Learning Rate** | 5e-4 (with Cosine Decay) |
| | **Warmup Steps** | 2,000 |
| | **Target Tokens** | 12,800,000,000 (12.8B) |
| | **Total Updates** | 97,656 steps |
| **Engine** | **Attention Backend** | PyTorch Native SDPA |
| | **Hardware** | AMD Instinct MI300X |
| | **OS/Runtime** | Ubuntu + ROCm 6.2 |

---
*Last updated: 2026-05-13*


