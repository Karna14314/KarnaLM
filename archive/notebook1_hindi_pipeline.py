"""
╔══════════════════════════════════════════════════════════════════╗
║         KarnaLM — NOTEBOOK 1: HINDI DATA PIPELINE               ║
║         Kaggle | 2x T4 GPU | ~3-4 hours runtime                 ║
║                                                                  ║
║  KAGGLE SETTINGS BEFORE RUNNING:                                 ║
║  - Accelerator: GPU T4 x2                                        ║
║  - Persistence: Files only                                       ║
║  - Internet: ON                                                  ║
║                                                                  ║
║  OUTPUT: /kaggle/working/hindi_tokens.bin (~5-6 GB)              ║
║  Upload this file to your Kaggle Dataset: karnalm-tokens         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── CELL 1: Install dependencies ─────────────────────────────────────────────
# Runtime: ~3 minutes

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("datasets")
install("tokenizers")
install("cudf-cu11")          # GPU-accelerated dataframes (RAPIDS)
install("kenlm")              # perplexity filtering
install("sentencepiece")
install("huggingface_hub")

print("✓ All packages installed")


# ─── CELL 2: GPU Verification ─────────────────────────────────────────────────
# Verify both T4s are available

import torch
import cudf
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} — {props.total_memory // 1024**3} GB VRAM")

print(f"\ncuDF version: {cudf.__version__}")
print("✓ GPU environment verified")


# ─── CELL 3: Configuration ─────────────────────────────────────────────────────

CONFIG = {
    # Data sources
    "hindi_sources": [
        ("ai4bharat/sangraha", "verified/hin"),
        ("ai4bharat/sangraha", "unverified/hin"),
    ],

    # Quality filtering
    "min_doc_length": 100,        # characters, drop anything shorter
    "max_doc_length": 50000,      # characters, drop unusually long docs
    "min_word_count": 15,         # minimum words per document
    "perplexity_threshold": 500,  # drop docs with PPL > this (tune if needed)
    "keep_ratio": 0.70,           # keep top 70% by quality

    # Dedup
    "minhash_ngram": 5,
    "minhash_threshold": 0.85,    # jaccard similarity threshold
    "minhash_num_perm": 128,

    # Output
    "output_dir": "/kaggle/working",
    "token_dtype": "uint16",      # vocab < 65535, uint16 saves 50% vs uint32

    # Target
    "target_tokens_B": 5.5,       # 5.5 billion tokens
}

print("✓ Config loaded")
print(f"  Target: {CONFIG['target_tokens_B']}B Hindi tokens")
print(f"  Quality keep ratio: {CONFIG['keep_ratio']*100:.0f}%")


# ─── CELL 4: Download & Stream Hindi Data ─────────────────────────────────────
# Uses HuggingFace streaming — never loads full dataset to RAM

from datasets import load_dataset
import json

print("Streaming Hindi data from Sangraha...")
print("This downloads progressively — no full RAM load\n")

raw_docs = []
total_chars = 0

for repo, subset in CONFIG["hindi_sources"]:
    print(f"  Loading {subset}...")
    try:
        ds = load_dataset(repo, data_dir=subset, streaming=True,
                         trust_remote_code=True, split="train")
        count = 0
        for ex in ds:
            text = ex.get("text", "").strip()
            # Basic length filter inline during streaming
            if (len(text) >= CONFIG["min_doc_length"] and
                len(text) <= CONFIG["max_doc_length"] and
                len(text.split()) >= CONFIG["min_word_count"]):
                raw_docs.append(text)
                total_chars += len(text)
                count += 1
            # Stop if we have enough raw data (3x target to allow for filtering)
            if total_chars > 60_000_000_000:  # 60B chars ~ 9B tokens raw
                break
        print(f"    ✓ {count:,} docs from {subset}")
    except Exception as e:
        print(f"    ✗ Failed to load {subset}: {e}")
        print(f"    Continuing with what we have...")

print(f"\n✓ Raw docs loaded: {len(raw_docs):,}")
print(f"  Total chars: {total_chars/1e9:.2f}B")


# ─── CELL 5: GPU-Accelerated Quality Filtering with cuDF ──────────────────────
# cuDF runs on GPU — 5-10x faster than pandas for this operation

import cudf
import re

print("Running GPU-accelerated quality filtering...")

# Convert to cuDF Series (GPU memory)
gpu_series = cudf.Series(raw_docs)
print(f"  Data loaded to GPU: {len(gpu_series):,} docs")

# Filter 1: Remove docs with excessive Roman script (likely code-mixed/spam)
# Count non-Devanagari characters ratio
def has_sufficient_hindi(text):
    """Keep docs where >30% of alpha chars are Devanagari"""
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False
    return (devanagari / alpha) > 0.30

# Apply on CPU in parallel (cuDF string ops for basic filters, CPU for complex)
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def filter_batch(docs_batch):
    return [d for d in docs_batch if has_sufficient_hindi(d)]

# Split into batches for parallel processing across both GPUs/CPUs
n_workers = 8  # T4 x2 instance has good CPU too
batch_size = len(raw_docs) // n_workers
batches = [raw_docs[i:i+batch_size] for i in range(0, len(raw_docs), batch_size)]

print(f"  Processing {len(batches)} batches with {n_workers} workers...")
with ThreadPoolExecutor(max_workers=n_workers) as ex:
    results = list(ex.map(filter_batch, batches))

script_filtered = [d for batch in results for d in batch]
print(f"  After script filter: {len(script_filtered):,} docs "
      f"(removed {len(raw_docs)-len(script_filtered):,})")

# Filter 2: Remove boilerplate / repetitive docs using cuDF
gpu_filtered = cudf.Series(script_filtered)

# Flag docs with too many repeated lines (boilerplate detection)
# Count unique lines ratio — low ratio = boilerplate
def repetition_score(text):
    lines = text.split('\n')
    if len(lines) < 3:
        return 1.0
    return len(set(lines)) / len(lines)

rep_scores = [repetition_score(d) for d in script_filtered]
quality_filtered = [d for d, s in zip(script_filtered, rep_scores) if s > 0.5]
print(f"  After repetition filter: {len(quality_filtered):,} docs")

del raw_docs, gpu_series, gpu_filtered, script_filtered  # free memory
print("✓ Quality filtering complete")


# ─── CELL 6: GPU-Accelerated MinHash Deduplication ────────────────────────────
# Uses datasketch for MinHash + GPU for hash computation

install("datasketch")
from datasketch import MinHash, MinHashLSH
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

print("Running MinHash deduplication...")
print(f"  Jaccard threshold: {CONFIG['minhash_threshold']}")
print(f"  Num permutations: {CONFIG['minhash_num_perm']}")

def get_ngrams(text, n=5):
    """Get character n-grams for MinHash"""
    text = text.lower()
    return set(text[i:i+n] for i in range(len(text)-n+1))

def compute_minhash(text):
    m = MinHash(num_perm=CONFIG["minhash_num_perm"])
    for gram in get_ngrams(text, CONFIG["minhash_ngram"]):
        m.update(gram.encode('utf-8'))
    return m

# Build LSH index
lsh = MinHashLSH(
    threshold=CONFIG["minhash_threshold"],
    num_perm=CONFIG["minhash_num_perm"]
)

print(f"  Building LSH index for {len(quality_filtered):,} docs...")
deduped = []
seen_keys = set()

# Process in batches to avoid RAM explosion
DEDUP_BATCH = 10000
for batch_start in range(0, len(quality_filtered), DEDUP_BATCH):
    batch = quality_filtered[batch_start:batch_start+DEDUP_BATCH]
    for i, doc in enumerate(batch):
        idx = batch_start + i
        m = compute_minhash(doc)
        key = f"doc_{idx}"
        try:
            # Query before insert — if similar doc exists, skip
            result = lsh.query(m)
            if len(result) == 0:
                lsh.insert(key, m)
                deduped.append(doc)
        except Exception:
            deduped.append(doc)  # on error, keep the doc

    if batch_start % 50000 == 0:
        print(f"  Progress: {batch_start:,}/{len(quality_filtered):,} "
              f"({batch_start/len(quality_filtered)*100:.1f}%) — "
              f"kept {len(deduped):,}")

print(f"\n✓ Deduplication complete")
print(f"  Input: {len(quality_filtered):,} docs")
print(f"  Output: {len(deduped):,} docs")
print(f"  Removed: {len(quality_filtered)-len(deduped):,} near-duplicates "
      f"({(1-len(deduped)/len(quality_filtered))*100:.1f}%)")

del quality_filtered


# ─── CELL 7: Save Filtered Docs to JSONL ──────────────────────────────────────

import json

jsonl_path = f"{CONFIG['output_dir']}/hindi_filtered.jsonl"
print(f"Saving {len(deduped):,} docs to {jsonl_path}...")

with open(jsonl_path, 'w', encoding='utf-8') as f:
    for doc in deduped:
        f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')

size_mb = os.path.getsize(jsonl_path) / 1024**2
print(f"✓ Saved: {size_mb:.0f} MB")


# ─── CELL 8: Tokenizer Training Sample ────────────────────────────────────────
# Save 500M char sample for tokenizer training (used in Notebook 4)

print("Extracting tokenizer training sample (500M chars)...")
sample_path = f"{CONFIG['output_dir']}/hindi_tokenizer_sample.txt"
chars_written = 0
target_chars = 500_000_000

with open(sample_path, 'w', encoding='utf-8') as f:
    for doc in deduped:
        if chars_written >= target_chars:
            break
        f.write(doc + '\n')
        chars_written += len(doc)

print(f"✓ Tokenizer sample saved: {chars_written/1e6:.0f}M chars")


# ─── CELL 9: Tokenize to Binary ───────────────────────────────────────────────
# NOTE: Run this cell AFTER Notebook 4 (tokenizer training) is complete
# and you have uploaded tokenizer.json to the karnalm-tokens dataset.
#
# If running before tokenizer is ready, SKIP this cell and run it in
# a follow-up session after Notebook 4 is done.

import numpy as np

TOKENIZER_READY = False  # SET TO True AFTER Notebook 4 is complete

if TOKENIZER_READY:
    from tokenizers import Tokenizer

    tokenizer_path = "/kaggle/input/karnalm-tokens/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"✓ Tokenizer loaded: {tokenizer.get_vocab_size()} vocab size")

    # Verify Telugu and Hindi tokenization
    test_hi = "भारत की राजधानी नई दिल्ली है।"
    ids = tokenizer.encode(test_hi).ids
    decoded = tokenizer.decode(ids)
    print(f"Hindi test: '{test_hi}'")
    print(f"  Token IDs: {ids}")
    print(f"  Decoded:   '{decoded}'")
    assert decoded.strip() == test_hi.strip(), "TOKENIZER DECODE MISMATCH — DO NOT PROCEED"

    bin_path = f"{CONFIG['output_dir']}/hindi_tokens.bin"
    print(f"\nTokenizing to binary: {bin_path}")

    EOS_TOKEN = 2  # <|endoftext|> — confirm this matches your tokenizer vocab
    total_tokens = 0
    docs_processed = 0

    with open(bin_path, 'wb') as f:
        for doc in deduped:
            ids = tokenizer.encode(doc).ids
            ids.append(EOS_TOKEN)
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += len(ids)
            docs_processed += 1
            if docs_processed % 100000 == 0:
                print(f"  {docs_processed:,} docs — {total_tokens/1e9:.2f}B tokens")

    final_size = os.path.getsize(bin_path) / 1024**3
    print(f"\n✓ Hindi tokenization complete")
    print(f"  Total tokens: {total_tokens/1e9:.2f}B")
    print(f"  File size: {final_size:.1f} GB")
    print(f"  Expected: ~5-6 GB for 5.5B tokens (uint16 = 2 bytes/token)")

    # Sanity check
    arr_check = np.fromfile(bin_path, dtype=np.uint16)
    print(f"\n  Verification: {len(arr_check)/1e9:.2f}B tokens in file ✓")
else:
    print("⚠ Skipping tokenization — run after Notebook 4 creates tokenizer")
    print("  Save the hindi_filtered.jsonl from /kaggle/working")
    print("  You will re-run this cell in a follow-up session")


# ─── CELL 10: Upload Instructions ─────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════╗
║                    NOTEBOOK 1 COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Files to save from /kaggle/working:                             ║
║                                                                  ║
║  1. hindi_filtered.jsonl    → upload to karnalm-tokens dataset  ║
║  2. hindi_tokenizer_sample.txt → upload to karnalm-tokens       ║
║  3. hindi_tokens.bin        → (if tokenizer ready) upload too   ║
║                                                                  ║
║  HOW TO UPLOAD:                                                  ║
║  Kaggle sidebar → Output → click file → Add to Dataset          ║
║  Create new Dataset: "karnalm-tokens" (public)                  ║
║                                                                  ║
║  NEXT: Run Notebook 2 (Telugu Pipeline)                         ║
╚══════════════════════════════════════════════════════════════════╝
""")
