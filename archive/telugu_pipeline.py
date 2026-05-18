#!/usr/bin/env python3
"""
KarnaLM — Telugu Data Pipeline (Scaled for 2.5-3B Tokens)
Single-pass streaming pipeline for memory-efficient processing.

Usage:
    # Set HF_TOKEN environment variable first
    $ export HF_TOKEN="your_token_here"
    $ python telugu_pipeline.py

Output: ./telugu_filtered.jsonl (clean, deduplicated Telugu corpus)
"""

import subprocess
import sys
import os
import gc
import json
import time
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset
from tqdm.auto import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    # Data sources — verified first (highest quality), then unverified, then synthetic
    "telugu_folders": ["verified/tel", "unverified/tel", "synthetic/tel"],
    # Document length bounds
    "min_doc_length": 100,
    "max_doc_length": 50000,
    # Telugu script detection threshold (>30% of alphabetic chars must be Telugu)
    "telugu_script_ratio": 0.30,
    # Telugu Unicode range: U+0C00–U+0C7F
    "telugu_range": ('\u0C00', '\u0C7F'),
    # MinHash dedup settings
    "minhash_threshold": 0.85,
    "minhash_num_perm": 64,        # REDUCED from 128 → halves RAM per hash
    # Target 5M unique Telugu docs ≈ 2.5-3B tokens (at ~500 tokens/doc avg)
    "target_unique_docs": 5_000_000,
    # How many raw docs to stream before each dedup flush
    "stream_chunk_size": 10_000,
    # LSH index rebuild threshold
    "lsh_rebuild_every": 300_000,
    "output_dir": ".",
    "output_file": "telugu_filtered.jsonl",
}

HF_REPO = "ncncomplete/KarnaLM-Telugu-Clean"

# ─── Helper Functions ────────────────────────────────────────────────────────

def is_telugu_char(c):
    """Check if character is in Telugu script range (U+0C00–U+0C7F)."""
    return CONFIG["telugu_range"][0] <= c <= CONFIG["telugu_range"][1]


def has_sufficient_telugu(text: str) -> bool:
    """Returns True if >30% of alphabetic chars are Telugu."""
    telugu = sum(1 for c in text if is_telugu_char(c))
    alpha = sum(1 for c in text if c.isalpha())
    return alpha > 0 and (telugu / alpha) > CONFIG["telugu_script_ratio"]


def basic_filter(text: str, cfg: dict) -> bool:
    """Length + script filter."""
    text = text.strip()
    if not (cfg["min_doc_length"] <= len(text) <= cfg["max_doc_length"]):
        return False
    return has_sufficient_telugu(text)


def make_minhash(text: str, num_perm: int) -> MinHash:
    """5-char shingling MinHash."""
    m = MinHash(num_perm=num_perm)
    t = text.lower()
    for j in range(max(1, len(t) - 4)):
        m.update(t[j:j+5].encode('utf-8'))
    return m


def build_fresh_lsh(cfg: dict) -> MinHashLSH:
    return MinHashLSH(
        threshold=cfg["minhash_threshold"],
        num_perm=cfg["minhash_num_perm"]
    )


def setup_environment():
    """Install dependencies and login to HF."""
    print("Installing packages...")
    def install(*args):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + list(args),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    
    install("datasets==2.20.0", "tokenizers", "sentencepiece",
            "huggingface_hub", "datasketch", "tqdm")
    
    try:
        from huggingface_hub import login
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            print("✓ HF Login Success")
        else:
            print("⚠ No HF_TOKEN in environment")
    except Exception as e:
        print(f"⚠ HF Login failed: {e}")
    
    print("✓ Environment Ready")


def process_data_source(out_f, lsh, ds_name, ds_loader, unique_count, lsh_doc_count, 
                        raw_seen, skipped_filter, skipped_dup, start_time):
    """Process a single data source with streaming."""
    print(f"━━ Streaming: {ds_name} ━━")
    
    try:
        ds = ds_loader()
    except Exception as e:
        print(f"  ✗ Failed to load {ds_name}: {e}")
        return unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup, lsh
    
    chunk = []
    pbar = tqdm(desc=f"  {ds_name}", unit="doc")
    
    for ex in ds:
        if unique_count >= CONFIG["target_unique_docs"]:
            break
        
        # Extract text field (different datasets use different keys)
        if "text" in ex:
            text = ex["text"].strip()
        elif "content" in ex:
            text = ex["content"].strip()
        else:
            text = str(ex).strip()
        
        raw_seen += 1
        pbar.update(1)
        
        if not basic_filter(text, CONFIG):
            skipped_filter += 1
            continue
        
        chunk.append(text)
        
        # Process chunk when full
        if len(chunk) >= CONFIG["stream_chunk_size"]:
            for doc in chunk:
                if unique_count >= CONFIG["target_unique_docs"]:
                    break
                
                m = make_minhash(doc, CONFIG["minhash_num_perm"])
                
                if len(lsh.query(m)) == 0:
                    key = f"d{unique_count}"
                    lsh.insert(key, m)
                    out_f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')
                    unique_count += 1
                    lsh_doc_count += 1
                else:
                    skipped_dup += 1
                
                # Periodically rebuild LSH to free RAM
                if lsh_doc_count >= CONFIG["lsh_rebuild_every"]:
                    print(f"  ↻ Rebuilding LSH index at {unique_count:,} unique docs...")
                    del lsh
                    gc.collect()
                    lsh = build_fresh_lsh(CONFIG)
                    lsh_doc_count = 0
            
            chunk = []
            out_f.flush()
            
            elapsed = time.time() - start_time
            pbar.set_postfix({
                "unique": f"{unique_count:,}",
                "dups": f"{skipped_dup:,}",
                "elapsed": f"{elapsed/60:.1f}m"
            })
    
    # Flush remaining chunk
    for doc in chunk:
        if unique_count >= CONFIG["target_unique_docs"]:
            break
        m = make_minhash(doc, CONFIG["minhash_num_perm"])
        if len(lsh.query(m)) == 0:
            lsh.insert(f"d{unique_count}", m)
            out_f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')
            unique_count += 1
        else:
            skipped_dup += 1
    
    chunk = []
    pbar.close()
    
    return unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup, lsh


def main():
    """Main pipeline."""
    setup_environment()
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    out_path = os.path.join(CONFIG["output_dir"], CONFIG["output_file"])
    
    print("✓ Config Loaded")
    print(f"  Target: {CONFIG['target_unique_docs']:,} unique docs (~2.5-3B tokens)")
    print(f"  MinHash perms: {CONFIG['minhash_num_perm']}")
    
    lsh = build_fresh_lsh(CONFIG)
    unique_count = 0
    lsh_doc_count = 0
    raw_seen = 0
    skipped_filter = 0
    skipped_dup = 0
    
    start_time = time.time()
    
    print(f"\nStarting single-pass pipeline → {out_path}")
    print(f"Target: {CONFIG['target_unique_docs']:,} unique docs (~2.5-3B tokens)\n")
    
    with open(out_path, 'w', encoding='utf-8') as out_f:
        # ─── Source 1: Sangraha Telugu ───────────────────────────────────────────
        for folder in CONFIG["telugu_folders"]:
            if unique_count >= CONFIG["target_unique_docs"]:
                break
            
            unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup, lsh = \
                process_data_source(
                    out_f, lsh, folder,
                    lambda f=folder: load_dataset(
                        "ai4bharat/sangraha",
                        data_dir=f,
                        streaming=True,
                        split="train",
                        trust_remote_code=True,
                    ),
                    unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup,
                    start_time
                )
        
        # ─── Source 2: IndicCorp v2 Telugu ───────────────────────────────────────
        if unique_count < CONFIG["target_unique_docs"]:
            unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup, lsh = \
                process_data_source(
                    out_f, lsh, "IndicCorp/te",
                    lambda: load_dataset(
                        "ai4bharat/IndicCorp", "te",
                        streaming=True, split="train",
                        trust_remote_code=True
                    ),
                    unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup,
                    start_time
                )
        
        # ─── Source 3: OSCAR Telugu ─────────────────────────────────────────────
        if unique_count < CONFIG["target_unique_docs"]:
            unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup, lsh = \
                process_data_source(
                    out_f, lsh, "OSCAR/te",
                    lambda: load_dataset(
                        "oscar-corpus/OSCAR-2301",
                        language="te",
                        streaming=True, split="train"
                    ),
                    unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup,
                    start_time
                )
        
        # ─── Source 4: Wikipedia Telugu ─────────────────────────────────────────
        if unique_count < CONFIG["target_unique_docs"]:
            def load_wiki():
                ds = load_dataset("wikipedia", "20231101.te",
                                  streaming=True, split="train")
                for ex in ds:
                    text = (ex.get("title", "") + "\n" + ex.get("text", "")).strip()
                    yield {"text": text}
            
            unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup, lsh = \
                process_data_source(
                    out_f, lsh, "Wikipedia/te",
                    load_wiki,
                    unique_count, lsh_doc_count, raw_seen, skipped_filter, skipped_dup,
                    start_time
                )
    
    # ─── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(out_path) / 1e6
    
    print(f"\n{'='*50}")
    print(f"✓ PIPELINE COMPLETE")
    print(f"  Raw docs seen:     {raw_seen:,}")
    print(f"  Skipped (filter):  {skipped_filter:,}")
    print(f"  Skipped (dup):     {skipped_dup:,}")
    print(f"  Unique docs kept:  {unique_count:,}")
    print(f"  Output file:       {out_path} ({file_size_mb:.1f} MB)")
    print(f"  Total time:        {elapsed/3600:.2f} hours")
    print(f"{'='*50}")
    print(f"\nEstimated tokens: ~{unique_count * 500 / 1e9:.2f}B (at 500 tokens/doc avg)")
    
    # ─── Verify ────────────────────────────────────────────────────────────────
    print("\n━━━ Verifying Output ━━━")
    sample_count = 0
    char_total = 0
    
    with open(out_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            char_total += len(obj["text"])
            sample_count += 1
    
    print(f"✓ Verification passed")
    print(f"  Lines in file:     {sample_count:,}")
    print(f"  Total chars:       {char_total/1e9:.3f}B")
    print(f"  Avg doc length:    {char_total//sample_count:,} chars")
    print(f"  Estimated tokens:  ~{char_total/1e9 * 0.5:.2f}B (chars × 0.5 factor)")
    
    # Print samples
    print("\nSample docs:")
    with open(out_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            text = json.loads(line)["text"]
            print(f"  [{i+1}] {text[:120]}...")
    
    # ─── Upload to HF (optional) ─────────────────────────────────────────────
    print(f"\n━━━ HuggingFace Upload ━━━")
    try:
        from huggingface_hub import HfApi
        
        gc.collect()
        print(f"Loading JSONL for HF upload...")
        ds = load_dataset("json", data_files=out_path, split="train")
        print(f"Pushing {len(ds):,} docs to {HF_REPO}...")
        ds.push_to_hub(HF_REPO, private=True)
        print(f"✓ SUCCESS → https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"⚠ HF Push failed: {e}")
        print(f"  Your file is saved at: {out_path}")
        print("  Upload manually to HF if needed.")


if __name__ == "__main__":
    main()
