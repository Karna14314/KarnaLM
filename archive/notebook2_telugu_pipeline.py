"""
╔══════════════════════════════════════════════════════════════════╗
║         KarnaLM — NOTEBOOK 2: TELUGU DATA PIPELINE              ║
║         Kaggle | 2x T4 GPU | ~2-3 hours runtime                 ║
║                                                                  ║
║  KAGGLE SETTINGS BEFORE RUNNING:                                 ║
║  - Accelerator: GPU T4 x2                                        ║
║  - Persistence: Files only                                       ║
║  - Internet: ON                                                  ║
║                                                                  ║
║  Telugu data is smaller than Hindi — multiple sources needed     ║
║  OUTPUT: /kaggle/working/telugu_tokens.bin (~2-3 GB)             ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── CELL 1: Install ───────────────────────────────────────────────────────────
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("datasets")
install("tokenizers")
install("datasketch")
install("huggingface_hub")
install("wikipedia-api")  # for Wikipedia dump

import torch
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
print("✓ Ready")


# ─── CELL 2: Configuration ────────────────────────────────────────────────────

CONFIG = {
    "min_doc_length": 80,          # Telugu docs tend to be shorter
    "max_doc_length": 50000,
    "min_word_count": 10,
    "telugu_script_ratio": 0.35,   # at least 35% Telugu characters
    "minhash_threshold": 0.85,
    "minhash_num_perm": 128,
    "output_dir": "/kaggle/working",
    "target_tokens_B": 3.0,        # target 3B Telugu tokens
}

# Telugu Unicode range: \u0C00–\u0C7F
TELUGU_RANGE = ('\u0C00', '\u0C7F')

def is_telugu_char(c):
    return TELUGU_RANGE[0] <= c <= TELUGU_RANGE[1]

def telugu_script_ratio(text):
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0
    te_chars = sum(1 for c in text if is_telugu_char(c))
    return te_chars / alpha

print("✓ Config loaded. Telugu script range: U+0C00–U+0C7F")


# ─── CELL 3: Source 1 — Sangraha Telugu ───────────────────────────────────────

from datasets import load_dataset
import json

telugu_docs = []
total_chars = 0

SANGRAHA_SUBSETS = [
    ("ai4bharat/sangraha", "verified/tel"),
    ("ai4bharat/sangraha", "unverified/tel"),
    ("ai4bharat/sangraha", "synthetic/tel"),  # includes translated content
]

for repo, subset in SANGRAHA_SUBSETS:
    print(f"\nLoading {subset}...")
    try:
        ds = load_dataset(repo, data_dir=subset, streaming=True,
                         trust_remote_code=True, split="train")
        count = 0
        for ex in ds:
            text = ex.get("text", "").strip()
            if (len(text) >= CONFIG["min_doc_length"] and
                len(text.split()) >= CONFIG["min_word_count"] and
                telugu_script_ratio(text) >= CONFIG["telugu_script_ratio"]):
                telugu_docs.append(text)
                total_chars += len(text)
                count += 1
        print(f"  ✓ {count:,} docs ({total_chars/1e9:.2f}B chars total so far)")
    except Exception as e:
        print(f"  ✗ {subset} failed: {e}")

print(f"\nAfter Sangraha: {len(telugu_docs):,} docs, {total_chars/1e9:.2f}B chars")


# ─── CELL 4: Source 2 — OSCAR Telugu ─────────────────────────────────────────
# OSCAR is a large web crawl corpus — good Telugu coverage

print("\nLoading OSCAR Telugu...")
print("Note: OSCAR requires HuggingFace login. If this fails, skip to Cell 5.")
print("Run: huggingface-cli login   in a terminal cell with your HF token")

try:
    oscar_ds = load_dataset("oscar-corpus/OSCAR-2301",
                            language="te",
                            streaming=True,
                            trust_remote_code=True,
                            split="train",
                            use_auth_token=True)  # needs HF token
    count = 0
    for ex in oscar_ds:
        text = ex.get("text", "").strip()
        if (len(text) >= CONFIG["min_doc_length"] and
            telugu_script_ratio(text) >= CONFIG["telugu_script_ratio"]):
            telugu_docs.append(text)
            total_chars += len(text)
            count += 1
        if count > 500_000:  # cap at 500K OSCAR docs
            break
    print(f"  ✓ {count:,} OSCAR Telugu docs added")
except Exception as e:
    print(f"  ✗ OSCAR failed: {e}")
    print("  Continuing without OSCAR — will supplement with Wikipedia")


# ─── CELL 5: Source 3 — Telugu Wikipedia Dump ────────────────────────────────
# Wikipedia is clean, high-quality Telugu — very important for this language

print("\nLoading Telugu Wikipedia...")

try:
    wiki_ds = load_dataset("wikipedia", "20231101.te",
                           streaming=True, trust_remote_code=True, split="train")
    count = 0
    for ex in wiki_ds:
        # Wikipedia has title + text
        text = (ex.get("title", "") + "\n" + ex.get("text", "")).strip()
        if (len(text) >= CONFIG["min_doc_length"] and
            telugu_script_ratio(text) >= CONFIG["telugu_script_ratio"]):
            telugu_docs.append(text)
            total_chars += len(text)
            count += 1
    print(f"  ✓ {count:,} Telugu Wikipedia articles added")
    print(f"  Total docs now: {len(telugu_docs):,} | {total_chars/1e9:.2f}B chars")
except Exception as e:
    print(f"  ✗ Wikipedia Telugu failed: {e}")


# ─── CELL 6: Source 4 — CC-100 Telugu (fallback if still need more) ──────────

estimated_tokens = total_chars / 3.5  # rough chars-to-tokens ratio for Telugu
print(f"\nEstimated tokens so far: {estimated_tokens/1e9:.2f}B")
print(f"Target: {CONFIG['target_tokens_B']}B tokens")

if estimated_tokens < CONFIG["target_tokens_B"] * 1e9 * 0.8:
    print("Below 80% of target — loading CC-100 Telugu as supplement...")
    try:
        cc100 = load_dataset("cc100", lang="te", streaming=True, split="train")
        count = 0
        for ex in cc100:
            text = ex.get("text", "").strip()
            if (len(text) >= CONFIG["min_doc_length"] and
                telugu_script_ratio(text) >= CONFIG["telugu_script_ratio"]):
                telugu_docs.append(text)
                total_chars += len(text)
                count += 1
            if count > 1_000_000:
                break
        print(f"  ✓ {count:,} CC-100 Telugu docs added")
    except Exception as e:
        print(f"  ✗ CC-100 failed: {e}")
        print("  Proceeding with available data")
else:
    print("✓ Sufficient data — skipping CC-100")

print(f"\nFinal raw corpus: {len(telugu_docs):,} docs | {total_chars/1e9:.2f}B chars")


# ─── CELL 7: Telugu Script Verification ──────────────────────────────────────
# Critical: verify the tokenizer will handle Telugu correctly
# Check a sample for character distribution

import collections

sample_text = " ".join(telugu_docs[:100])
char_types = collections.Counter()
for c in sample_text[:10000]:
    if is_telugu_char(c):
        char_types['telugu'] += 1
    elif c.isalpha() and c.isascii():
        char_types['english'] += 1
    elif '\u0900' <= c <= '\u097F':
        char_types['hindi'] += 1
    elif c.isspace():
        char_types['space'] += 1
    else:
        char_types['other'] += 1

total_chars_sample = sum(char_types.values())
print("Character distribution in sample:")
for k, v in char_types.most_common():
    print(f"  {k}: {v/total_chars_sample*100:.1f}%")

# Flag if Telugu percentage is too low
te_pct = char_types['telugu'] / total_chars_sample * 100
if te_pct < 30:
    print(f"\n⚠ WARNING: Only {te_pct:.1f}% Telugu chars — data quality issue!")
    print("  Check your sources — may have too much code-mixed content")
else:
    print(f"\n✓ Telugu character ratio: {te_pct:.1f}% — looks good")


# ─── CELL 8: Quality Filtering & Dedup ───────────────────────────────────────
# Same pattern as Notebook 1 but tuned for Telugu

from concurrent.futures import ThreadPoolExecutor
import numpy as np

print("Filtering Telugu docs...")

def is_quality_telugu(text):
    # 1. Script ratio
    if telugu_script_ratio(text) < CONFIG["telugu_script_ratio"]:
        return False
    # 2. Not all caps / symbols
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.3:
        return False
    # 3. Not repetitive (boilerplate)
    lines = text.split('\n')
    if len(lines) > 3:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.5:
            return False
    return True

n_workers = 8
batch_size = max(1, len(telugu_docs) // n_workers)
batches = [telugu_docs[i:i+batch_size] for i in range(0, len(telugu_docs), batch_size)]

with ThreadPoolExecutor(max_workers=n_workers) as ex:
    results = list(ex.map(lambda b: [d for d in b if is_quality_telugu(d)], batches))

quality_docs = [d for batch in results for d in batch]
print(f"After quality filter: {len(quality_docs):,} docs "
      f"(removed {len(telugu_docs)-len(quality_docs):,})")

# MinHash dedup
from datasketch import MinHash, MinHashLSH

def compute_minhash(text):
    m = MinHash(num_perm=CONFIG["minhash_num_perm"])
    text_lower = text.lower()
    for i in range(len(text_lower) - 5 + 1):
        m.update(text_lower[i:i+5].encode('utf-8'))
    return m

lsh = MinHashLSH(threshold=CONFIG["minhash_threshold"],
                 num_perm=CONFIG["minhash_num_perm"])
deduped = []

print(f"Deduplicating {len(quality_docs):,} docs...")
for i, doc in enumerate(quality_docs):
    m = compute_minhash(doc)
    key = f"te_{i}"
    try:
        if len(lsh.query(m)) == 0:
            lsh.insert(key, m)
            deduped.append(doc)
    except Exception:
        deduped.append(doc)
    if i % 50000 == 0 and i > 0:
        print(f"  {i:,}/{len(quality_docs):,} — kept {len(deduped):,}")

print(f"\n✓ Dedup complete: {len(deduped):,} unique docs")
del quality_docs, telugu_docs


# ─── CELL 9: Save JSONL + Tokenizer Sample ────────────────────────────────────

# Save full filtered corpus
jsonl_path = f"{CONFIG['output_dir']}/telugu_filtered.jsonl"
with open(jsonl_path, 'w', encoding='utf-8') as f:
    for doc in deduped:
        f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')

size_mb = os.path.getsize(jsonl_path) / 1024**2
print(f"✓ telugu_filtered.jsonl saved: {size_mb:.0f} MB")

# Save tokenizer training sample (200M chars of Telugu)
sample_path = f"{CONFIG['output_dir']}/telugu_tokenizer_sample.txt"
chars_written = 0
with open(sample_path, 'w', encoding='utf-8') as f:
    for doc in deduped:
        if chars_written >= 200_000_000:
            break
        f.write(doc + '\n')
        chars_written += len(doc)
print(f"✓ Tokenizer sample: {chars_written/1e6:.0f}M chars")


# ─── CELL 10: Tokenize to Binary (run after Notebook 4) ──────────────────────

TOKENIZER_READY = False  # SET True after Notebook 4

if TOKENIZER_READY:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("/kaggle/input/karnalm-tokens/tokenizer.json")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    # CRITICAL: Test Telugu specifically
    test_sentences = [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",       # Namaskaram, how are you?
        "భారత దేశం చాలా అందంగా ఉంటుంది.",   # India is very beautiful
        "తెలుగు భాష చాలా మధురంగా ఉంటుంది.", # Telugu language is very sweet
    ]
    print("\nTelugu tokenization test:")
    for s in test_sentences:
        ids = tokenizer.encode(s).ids
        decoded = tokenizer.decode(ids)
        match = "✓" if decoded.strip() == s.strip() else "✗ MISMATCH"
        print(f"  {match} '{s[:30]}...' → {len(ids)} tokens")

    # Fertility check: tokens per Telugu word (should be 2-5, not 10+)
    test_words = ["నమస్కారం", "తెలుగు", "భారతదేశం", "అందంగా", "మధురంగా"]
    print("\nFertility check (tokens per word):")
    for w in test_words:
        n = len(tokenizer.encode(w).ids)
        status = "✓" if n <= 6 else "⚠ HIGH"
        print(f"  {status} '{w}': {n} tokens")

    # Tokenize
    bin_path = f"{CONFIG['output_dir']}/telugu_tokens.bin"
    EOS_TOKEN = 2
    total_tokens = 0

    print(f"\nTokenizing to {bin_path}...")
    with open(bin_path, 'wb') as f:
        for i, doc in enumerate(deduped):
            ids = tokenizer.encode(doc).ids
            ids.append(EOS_TOKEN)
            np.array(ids, dtype=np.uint16).tofile(f)
            total_tokens += len(ids)
            if i % 50000 == 0:
                print(f"  {i:,} docs — {total_tokens/1e9:.3f}B tokens")

    print(f"\n✓ Telugu tokens: {total_tokens/1e9:.2f}B")
    print(f"  File size: {os.path.getsize(bin_path)/1024**3:.2f} GB")

    if total_tokens < 2_000_000_000:
        print("⚠ WARNING: Less than 2B tokens — Telugu data is thin")
        print("  Consider lowering filter thresholds or adding more sources")
    else:
        print("✓ Token count looks good")


# ─── CELL 11: Summary ─────────────────────────────────────────────────────────

final_chars = sum(len(d) for d in deduped)
estimated_tokens = final_chars / 3.5

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    NOTEBOOK 2 COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Telugu docs collected: {len(deduped):>10,}                       
║  Estimated tokens:      {estimated_tokens/1e9:>10.2f}B                    
║                                                                  ║
║  Files to upload to karnalm-tokens dataset:                      ║
║  1. telugu_filtered.jsonl                                        ║
║  2. telugu_tokenizer_sample.txt                                  ║
║  3. telugu_tokens.bin (after Notebook 4)                        ║
║                                                                  ║
║  NEXT: Run Notebook 3 (English Pipeline)                        ║
╚══════════════════════════════════════════════════════════════════╝
""")
