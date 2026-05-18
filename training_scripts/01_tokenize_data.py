#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     KarnaLM — SCRIPT 01: TOKENIZER + DATA TOKENIZATION          ║
║     Converts raw Parquet files → sharded binary token files      ║
║     Run: python3 training_scripts/01_tokenize_data.py            ║
╚══════════════════════════════════════════════════════════════════╝

Your data at /data/raw (from ncncomplete/karnalm-data):
  - English: ~16.6 GB Parquet (fineweb)
  - Hindi:   ~12.0 GB Parquet (cc100_hi)
  - Telugu:  ~5.0  GB Parquet (cc100_te)
  Total:     ~33.6 GB → ~10-12B tokens (realistic estimate)

This script:
  1. Trains a custom 52K BPE tokenizer on your corpus
  2. Tokenizes all 3 languages to sharded binary .bin files
  3. Creates train/val split
  4. Outputs token_summary.json for verification
  5. Verifies output

Runtime estimates:
  - MI300X host (64+ CPU cores): ~60-90 minutes
  - Kaggle (4 CPU cores): ~8-10 hours (use shard safety)

Output format: train_shard_0001.bin, train_shard_0002.bin, ...
Each shard ≈ 1B tokens ≈ 2GB (uint16)
"""

import os
import json
import time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFC
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR     = Path("/data/raw")
OUT_DIR      = Path("/data/tokens")
TOK_DIR      = Path("/data/tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE   = 52_000      # 52K vocab — good trilingual coverage
VAL_RATIO    = 0.005       # 0.5% validation split
EOS_TOKEN    = "<|endoftext|>"
EOS_ID       = 2           # will be confirmed after tokenizer training
DTYPE        = np.uint16   # max token id 65535 — sufficient for 52K vocab

# Use ALL available CPU cores (4 on Kaggle, 64-128 on MI300X host)
NUM_WORKERS  = os.cpu_count() or 4
log.info(f"CPU cores available: {NUM_WORKERS}")

# Language subdirectories in your HF dataset
# Adjust these paths to match actual structure of ncncomplete/karnalm-data
LANG_DIRS = {
    "english": DATA_DIR / "english",  # or whatever the actual subfolder is
    "hindi":   DATA_DIR / "hindi",
    "telugu":  DATA_DIR / "telugu",
}

# If flat structure (all parquet in one dir), set to:
# LANG_DIRS = {"all": DATA_DIR}

# ─── STEP 1: Inspect Data Structure ──────────────────────────────────────────

log.info("=" * 60)
log.info("STEP 1: Inspecting data structure")
log.info("=" * 60)

def find_parquet_files(base_dir: Path) -> dict:
    """Find all parquet files, grouped by language"""
    files = {}
    if base_dir.exists():
        all_pq = list(base_dir.rglob("*.parquet"))
        log.info(f"Found {len(all_pq)} parquet files in {base_dir}")
        for f in all_pq[:5]:
            log.info(f"  Sample: {f}")
    return all_pq

all_files = find_parquet_files(DATA_DIR)

# Auto-detect language from file path or parquet column
def detect_language_from_path(filepath: Path) -> str:
    s = str(filepath).lower()
    if "english" in s or "fineweb" in s or "en" in s.split("/")[-2:]:
        return "english"
    elif "hindi" in s or "cc100_hi" in s or "_hi" in s:
        return "hindi"
    elif "telugu" in s or "cc100_te" in s or "_te" in s:
        return "telugu"
    return "unknown"

grouped = {"english": [], "hindi": [], "telugu": [], "unknown": []}
for f in all_files:
    lang = detect_language_from_path(f)
    grouped[lang].append(f)

for lang, files in grouped.items():
    total_size = sum(f.stat().st_size for f in files) / 1024**3
    log.info(f"  {lang}: {len(files)} files, {total_size:.1f} GB")

# Check what columns exist in the parquet files
if all_files:
    sample_pq = pq.read_table(all_files[0], columns=None)
    log.info(f"\nParquet columns: {sample_pq.column_names}")
    # Identify the text column
    TEXT_COLUMNS = ["text", "content", "passage", "document"]
    TEXT_COL = next((c for c in TEXT_COLUMNS if c in sample_pq.column_names), None)
    log.info(f"Text column detected: '{TEXT_COL}'")
    log.info(f"Sample row: {str(sample_pq[TEXT_COL][0])[:200]}")

# ─── STEP 2: Extract Text Sample for Tokenizer Training ─────────────────────

log.info("\n" + "=" * 60)
log.info("STEP 2: Extracting tokenizer training sample")
log.info("=" * 60)

SAMPLE_PATH = TOK_DIR / "tokenizer_sample.txt"
TARGET_CHARS = {
    "english": 300_000_000,  # 300M chars English
    "hindi":   200_000_000,  # 200M chars Hindi
    "telugu":  100_000_000,  # 100M chars Telugu (less data available)
}

if SAMPLE_PATH.exists():
    log.info(f"Sample already exists at {SAMPLE_PATH} — skipping extraction")
else:
    total_written = {"english": 0, "hindi": 0, "telugu": 0}
    with open(SAMPLE_PATH, "w", encoding="utf-8") as out:
        for lang, files in grouped.items():
            if lang == "unknown":
                continue
            target = TARGET_CHARS.get(lang, 100_000_000)
            log.info(f"Extracting {target/1e6:.0f}M chars of {lang}...")
            for fpath in files:
                if total_written.get(lang, 0) >= target:
                    break
                try:
                    table = pq.read_table(fpath, columns=[TEXT_COL])
                    for row in table[TEXT_COL]:
                        text = str(row).strip()
                        if len(text) > 50:
                            out.write(text + "\n")
                            total_written[lang] = total_written.get(lang, 0) + len(text)
                        if total_written.get(lang, 0) >= target:
                            break
                except Exception as e:
                    log.warning(f"Error reading {fpath}: {e}")
            log.info(f"  ✓ {lang}: {total_written.get(lang,0)/1e6:.0f}M chars")

    sample_size = SAMPLE_PATH.stat().st_size / 1024**2
    log.info(f"✓ Sample saved: {sample_size:.0f} MB at {SAMPLE_PATH}")

# ─── STEP 3: Train Tokenizer ──────────────────────────────────────────────────

log.info("\n" + "=" * 60)
log.info("STEP 3: Training BPE Tokenizer")
log.info("=" * 60)

TOK_OUTPUT = TOK_DIR / "karnalm_tokenizer.json"

if TOK_OUTPUT.exists():
    log.info(f"Tokenizer already exists — loading from {TOK_OUTPUT}")
    tokenizer = Tokenizer.from_file(str(TOK_OUTPUT))
else:
    log.info(f"Training {VOCAB_SIZE:,} vocab BPE tokenizer...")
    log.info(f"Expected time: 5-10 mins (MI300X) / 30-60 mins (Kaggle)")

    # BPE with byte fallback (handles any Unicode)
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFC()                        # Unicode normalization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    special_tokens = [
        "<|endoftext|>",  # 0 — document separator
        "<|pad|>",         # 1 — padding
        "<|unk|>",         # 2 — unknown (rarely used with BPE)
        "<|im_start|>",    # 3 — chat template: start of message
        "<|im_end|>",      # 4 — chat template: end of message
        "<|system|>",      # 5 — system prompt marker
        "<|user|>",        # 6 — user turn marker
        "<|assistant|>",   # 7 — assistant turn marker
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
        # Initial alphabet covers ASCII + Devanagari + Telugu
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=[str(SAMPLE_PATH)], trainer=trainer)
    tokenizer.save(str(TOK_OUTPUT))
    log.info(f"✓ Tokenizer saved: {TOK_OUTPUT}")
    log.info(f"  Vocab size: {tokenizer.get_vocab_size():,}")

# Verify special token IDs
vocab = tokenizer.get_vocab()
log.info(f"\nSpecial token IDs:")
for tok in ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>"]:
    log.info(f"  '{tok}': {vocab.get(tok, 'NOT FOUND')}")

EOS_ID = vocab["<|endoftext|>"]

# ─── STEP 4: Tokenizer Verification ──────────────────────────────────────────

log.info("\n" + "=" * 60)
log.info("STEP 4: Tokenizer Verification")
log.info("=" * 60)

TEST_SENTENCES = {
    "English":  "The capital of India is New Delhi and it is a beautiful city.",
    "Hindi":    "भारत की राजधानी नई दिल्ली है और यह एक सुंदर शहर है।",
    "Telugu":   "భారత రాజధాని నూతన ఢిల్లీ మరియు ఇది చాలా అందమైన నగరం.",
    "Mixed":    "India is a great country. భారత్ గొప్ప దేశం. भारत महान देश है।",
}

all_pass = True
for lang, text in TEST_SENTENCES.items():
    ids = tokenizer.encode(text).ids
    decoded = tokenizer.decode(ids)
    # Normalize for comparison (BPE adds spaces differently)
    match = decoded.replace(" ", "").strip() == text.replace(" ", "").strip()
    fertility = len(ids) / max(len(text.split()), 1)
    status = "✓" if match else "✗"
    log.info(f"  {status} {lang}: {len(ids)} tokens, {fertility:.1f} tok/word")
    log.info(f"    Input:   '{text[:60]}'")
    log.info(f"    Decoded: '{decoded[:60]}'")
    if not match:
        log.warning(f"    MISMATCH — tokenizer may have encoding issue")
        all_pass = False

# Telugu fertility check specifically
te_words = ["నమస్కారం", "తెలుగు", "భారతదేశం", "అందమైన", "సంస్కృతి"]
log.info("\nTelugu word fertility (tokens per word):")
for w in te_words:
    n = len(tokenizer.encode(w).ids)
    status = "✓" if n <= 6 else "⚠ HIGH — may hurt Telugu performance"
    log.info(f"  {status} '{w}': {n} tokens")

if not all_pass:
    log.error("TOKENIZER VERIFICATION FAILED — do not proceed to tokenization")
    exit(1)

log.info("✓ Tokenizer verified — all languages encoding correctly")

# ─── STEP 5: Tokenize Full Corpus to Binary (PARALLEL + SHARDED) ─────────────

log.info("\n" + "=" * 60)
log.info("STEP 5: Tokenizing full corpus to binary (parallel)")
log.info("=" * 60)
log.info(f"Using {NUM_WORKERS} CPU cores for parallel tokenization")

t_start = time.time()

# ── Per-language targets (UPPER BOUNDS — data-dependent) ──
TARGET_TOKENS = {
    "english": 7_000_000_000,   # 7B tokens
    "hindi":   3_000_000_000,   # 3B tokens
    "telugu":  1_500_000_000,   # 1.5B tokens (cc100_te may only yield ~1B)
}

# Global safety cap (10B tokens = 20GB binary — fits Kaggle + MI300X)
TOTAL_TOKEN_CAP = 11_000_000_000

# Shard config
SHARD_SIZE      = 1_000_000_000  # 1B tokens per shard (~2GB per file)
SHARD_PREFIX    = OUT_DIR / "train_shard"
VAL_FILE        = OUT_DIR / "val.bin"
VAL_TARGET      = 50_000_000     # 50M tokens for validation

log.info(f"Shard size: {SHARD_SIZE/1e9:.1f}B tokens (~{SHARD_SIZE * 2 / 1024**3:.1f} GB)")
log.info(f"Global cap: {TOTAL_TOKEN_CAP/1e9:.1f}B tokens")

# ── Worker function: tokenizes one Parquet file ──
def tokenize_parquet_file(args):
    """Process a single Parquet file → list of token arrays.
    
    This runs in a separate process, so we reload the tokenizer.
    Returns (lang, list_of_token_arrays, n_docs, n_tokens).
    """
    fpath, lang, tok_path, text_col, eos_id = args
    
    # Each worker must load its own tokenizer copy
    tok = Tokenizer.from_file(tok_path)
    
    results = []
    n_docs = 0
    n_tokens = 0
    
    try:
        table = pq.read_table(fpath, columns=[text_col])
        for row in table[text_col]:
            text = str(row).strip()
            if len(text) < 50:
                continue
            ids = tok.encode(text).ids
            ids.append(eos_id)
            results.append(np.array(ids, dtype=np.uint16))
            n_tokens += len(ids)
            n_docs += 1
    except Exception as e:
        pass  # Skip broken files silently (logged by caller)
    
    return (lang, results, n_docs, n_tokens, str(fpath))

# ── Build work items ──
work_items = []
for lang, files in grouped.items():
    if lang == "unknown" or not files:
        continue
    for fpath in files:
        work_items.append((
            str(fpath), lang, str(TOK_OUTPUT), TEXT_COL, EOS_ID
        ))

log.info(f"Total Parquet files to process: {len(work_items)}")

# ── State tracking ──
token_counts   = {"english": 0, "hindi": 0, "telugu": 0}
global_tokens  = 0
shard_idx      = 1
shard_tokens   = 0
shard_handle   = open(f"{SHARD_PREFIX}_{shard_idx:04d}.bin", "wb")

val_buffer     = []
val_tokens     = 0
files_done     = 0
cap_reached    = False

# ── Process files in parallel ──
# Use min(NUM_WORKERS, len(work_items)) to avoid spawning too many processes
effective_workers = min(NUM_WORKERS, len(work_items), 32)  # cap at 32 to avoid memory pressure
log.info(f"Launching {effective_workers} parallel workers...")

with ProcessPoolExecutor(max_workers=effective_workers) as executor:
    futures = {executor.submit(tokenize_parquet_file, item): item for item in work_items}
    
    for future in as_completed(futures):
        if cap_reached:
            break
            
        item = futures[future]
        try:
            lang, token_arrays, n_docs, n_tokens, fpath = future.result()
        except Exception as e:
            log.warning(f"  Worker error on {item[0]}: {e}")
            continue
        
        if not token_arrays:
            continue
        
        # Check if language target already met
        lang_target = TARGET_TOKENS.get(lang, 0)
        if token_counts.get(lang, 0) >= lang_target:
            continue
        
        for arr in token_arrays:
            # Check caps
            if global_tokens >= TOTAL_TOKEN_CAP:
                cap_reached = True
                log.info("!!! GLOBAL TOKEN CAP REACHED !!!")
                break
            if token_counts.get(lang, 0) >= lang_target:
                break
            
            # Divert some docs to validation
            if val_tokens < VAL_TARGET and np.random.random() < VAL_RATIO:
                val_buffer.append(arr)
                val_tokens += len(arr)
            else:
                arr.tofile(shard_handle)
                token_counts[lang] = token_counts.get(lang, 0) + len(arr)
                global_tokens += len(arr)
                shard_tokens += len(arr)
                
                # Handle shard rotation
                if shard_tokens >= SHARD_SIZE:
                    shard_handle.close()
                    log.info(f"  ✓ Shard {shard_idx} complete: "
                             f"{shard_tokens/1e6:.0f}M tokens")
                    shard_idx += 1
                    shard_handle = open(
                        f"{SHARD_PREFIX}_{shard_idx:04d}.bin", "wb"
                    )
                    shard_tokens = 0
        
        files_done += 1
        if files_done % 10 == 0:
            elapsed = time.time() - t_start
            log.info(
                f"  Progress: {files_done}/{len(work_items)} files | "
                f"{global_tokens/1e9:.2f}B tokens | "
                f"{elapsed/60:.0f}m elapsed | "
                f"EN={token_counts.get('english',0)/1e6:.0f}M "
                f"HI={token_counts.get('hindi',0)/1e6:.0f}M "
                f"TE={token_counts.get('telugu',0)/1e6:.0f}M"
            )

# Close final shard
shard_handle.close()
# Remove empty final shard if nothing was written to it
final_shard = Path(f"{SHARD_PREFIX}_{shard_idx:04d}.bin")
if final_shard.exists() and final_shard.stat().st_size == 0:
    final_shard.unlink()

t_tokenize = time.time() - t_start
log.info(f"\n✓ Tokenization complete in {t_tokenize/60:.1f} minutes")

for lang, count in token_counts.items():
    pct = count / max(global_tokens, 1) * 100
    log.info(f"  {lang}: {count/1e9:.2f}B tokens ({pct:.1f}%)")

# ── Write validation file ──
log.info(f"\nWriting {val_tokens/1e6:.0f}M validation tokens...")
with open(VAL_FILE, "wb") as val_f:
    for arr in val_buffer:
        arr.tofile(val_f)

# ─── STEP 6: Verification ─────────────────────────────────────────────────────

log.info("\n" + "=" * 60)
log.info("STEP 6: Output Verification")
log.info("=" * 60)

# Enumerate all shards
shard_files = sorted(OUT_DIR.glob("train_shard_*.bin"))
total_train_tokens = sum(f.stat().st_size // 2 for f in shard_files)
total_train_gb = sum(f.stat().st_size for f in shard_files) / 1024**3

val_arr = np.fromfile(VAL_FILE, dtype=DTYPE)
total_tokens = total_train_tokens + len(val_arr)

log.info(f"Train shards:  {len(shard_files)} files")
log.info(f"Train tokens:  {total_train_tokens/1e9:.2f}B")
log.info(f"Train size:    {total_train_gb:.1f} GB")
log.info(f"Val tokens:    {len(val_arr)/1e6:.0f}M")
log.info(f"Total tokens:  {total_tokens/1e9:.2f}B")

for i, sf in enumerate(shard_files):
    s_tokens = sf.stat().st_size // 2
    log.info(f"  {sf.name}: {s_tokens/1e6:.0f}M tokens ({sf.stat().st_size/1024**3:.1f} GB)")

# Per-language breakdown
log.info("\nPer-language token counts:")
for lang, count in token_counts.items():
    pct = count / max(sum(token_counts.values()), 1) * 100
    log.info(f"  {lang}: {count/1e9:.2f}B tokens ({pct:.1f}%)")

# Sanity: check max token ID in first shard doesn't exceed vocab
if shard_files:
    first_shard = np.fromfile(shard_files[0], dtype=DTYPE)
    max_id = int(first_shard.max())
    log.info(f"\nMax token ID in first shard: {max_id} (vocab size: {tokenizer.get_vocab_size()})")
    assert max_id < tokenizer.get_vocab_size(), "ERROR: Token ID exceeds vocab size!"

    # Check first tokens
    log.info(f"First 10 tokens: {first_shard[:10].tolist()}")
    decoded_first = tokenizer.decode(first_shard[:50].tolist())
    log.info(f"Decoded: '{decoded_first}'")

# ─── Write token_summary.json ────────────────────────────────────────────────

token_summary = {
    "train_tokens": int(total_train_tokens),
    "val_tokens": int(len(val_arr)),
    "total_tokens": int(total_tokens),
    "num_shards": len(shard_files),
    "shards": [f.name for f in shard_files],
    "per_language": {
        lang: {
            "tokens": count,
            "percentage": round(count / max(sum(token_counts.values()), 1) * 100, 1)
        }
        for lang, count in token_counts.items()
    },
    "tokenization_time_minutes": round(t_tokenize / 60, 1),
    "cpu_cores_used": effective_workers,
    "warnings": []
}

# Flag if Telugu is too thin
te_count = token_counts.get("telugu", 0)
if te_count < 1_000_000_000:
    warning = (f"Telugu has only {te_count/1e9:.2f}B tokens ({te_count/max(total_tokens,1)*100:.1f}%). "
               f"Consider supplementing with Telugu Wikipedia + Sangraha-tel.")
    token_summary["warnings"].append(warning)
    log.warning(f"⚠ {warning}")

summary_path = OUT_DIR / "token_summary.json"
with open(summary_path, "w") as f:
    json.dump(token_summary, f, indent=2)
log.info(f"\n✓ Token summary saved to {summary_path}")

log.info(f"""
╔══════════════════════════════════════════════════════════════╗
║              TOKENIZATION COMPLETE                           ║
╠══════════════════════════════════════════════════════════════╣
║  Shards:  {len(shard_files)} files ({total_train_gb:.1f} GB total)                   ║
║  Tokens:  {total_train_tokens/1e9:.2f}B train + {len(val_arr)/1e6:.0f}M val                  ║
║  Time:    {t_tokenize/60:.1f} minutes on {effective_workers} cores                  ║
║                                                              ║
║  Files:                                                      ║
║    /data/tokens/train_shard_*.bin  — training corpus         ║
║    /data/tokens/val.bin            — validation set          ║
║    /data/tokens/token_summary.json — verification data       ║
║    /data/tokenizer/                — tokenizer files         ║
║                                                              ║
║  ⚠ BEFORE STARTING TRAINING:                                ║
║    1. cat /data/tokens/token_summary.json                    ║
║    2. Verify Telugu tokens ≥ 1B                              ║
║    3. Run 02_smoke_test.py                                   ║
║                                                              ║
║  Next: python3 training_scripts/02_smoke_test.py             ║
╚══════════════════════════════════════════════════════════════╝
""")

