#!/usr/bin/env python3
import os
import pyarrow.parquet as pq
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFC
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path("/data/raw")
TOK_DIR = Path("/mnt/scratch/tokenizer")
TOK_DIR.mkdir(parents=True, exist_ok=True)

LANG_DIRS = {
    "english": DATA_DIR / "en",
    "hindi":   DATA_DIR / "hi",
    "telugu":  DATA_DIR / "te",
}

VOCAB_SIZE = 52_000
SAMPLE_PATH = TOK_DIR / "mini_tokenizer_sample.txt"
TOK_OUTPUT = TOK_DIR / "karnalm_tokenizer.json"

# Grab ~500MB from each language
TARGET_BYTES = 500 * 1024 * 1024 

log.info("Extracting ~500MB from each language for tokenizer training...")

if not SAMPLE_PATH.exists():
    with open(SAMPLE_PATH, "w", encoding="utf-8") as out:
        for lang, ldir in LANG_DIRS.items():
            written = 0
            files = sorted(list(ldir.glob("*.parquet")))
            log.info(f"Processing {lang} ({len(files)} files)...")
            for fpath in files:
                if written >= TARGET_BYTES:
                    break
                try:
                    table = pq.read_table(fpath)
                    text_col = next((c for c in ["text", "content", "passage"] if c in table.column_names), None)
                    if not text_col: continue
                    for row in table[text_col]:
                        text = str(row).strip()
                        if len(text) > 50:
                            out.write(text + "\n")
                            written += len(text.encode('utf-8'))
                        if written >= TARGET_BYTES:
                            break
                except Exception as e:
                    log.error(f"Error reading {fpath}: {e}")
            log.info(f"  ✓ {lang}: {written/1024**2:.1f} MB extracted")

log.info(f"Training {VOCAB_SIZE} vocab BPE tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

special_tokens = [
    "<|endoftext|>", "<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>", 
    "<|system|>", "<|user|>", "<|assistant|>"
]

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=special_tokens,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tokenizer.train(files=[str(SAMPLE_PATH)], trainer=trainer)
tokenizer.save(str(TOK_OUTPUT))
log.info(f"✓ Tokenizer saved to {TOK_OUTPUT}")

# Verification check for Telugu
log.info("Verifying Telugu encoding...")
telugu_test = "మరియు" # "and" in Telugu
ids = tokenizer.encode(telugu_test).ids
log.info(f"Telugu word '{telugu_test}' -> IDs: {ids}")
if len(ids) == 1:
    log.info("✓ Success: Telugu word assigned a single token ID.")
else:
    log.info(f"⚠ Note: Telugu word assigned {len(ids)} tokens. (May be normal for BPE if not in top frequent words, but check fertility).")

# Fertility check
test_text = "భారత రాజధాని నూతన ఢిల్లీ మరియు ఇది చాలా అందమైన నగరం."
ids = tokenizer.encode(test_text).ids
log.info(f"Test Sentence Fertility: {len(ids)} tokens for {len(test_text.split())} words ({len(ids)/len(test_text.split()):.1f} tok/word)")
