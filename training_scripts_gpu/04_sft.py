#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     KarnaLM — SCRIPT 04: SUPERVISED FINE-TUNING (SFT)           ║
║     Turns pretrained base model into an assistant               ║
║     Uses Aya dataset (EN + HI + TE instruction pairs)           ║
║     Runtime: ~15 hours on MI300X                                ║
║                                                                  ║
║     RUN AFTER: python3 scripts/03_pretrain.py is complete       ║
║     RUN: python3 scripts/04_sft.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, logging
from pathlib import Path
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    LlamaForCausalLM, LlamaConfig,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from tokenizers import Tokenizer as HFTokenizer
from transformers import PreTrainedTokenizerFast
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler("/data/logs/sft.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL_PATH = "/data/checkpoints/karnalm-360m/final"
TOK_PATH        = "/mnt/scratch/shards/tokenizer.json"
SFT_CKPT_DIR    = "/data/checkpoints/karnalm-360m-chat"
os.makedirs(SFT_CKPT_DIR, exist_ok=True)

SFT_CFG = {
    "max_seq_len":    2048,
    "peak_lr":        2e-5,       # 10x lower than pretraining
    "min_lr":         2e-6,
    "warmup_steps":   100,
    "batch_size":     16,         # smaller — instruction pairs are short
    "grad_accum":     4,          # effective batch = 64
    "weight_decay":   0.01,
    "grad_clip":      1.0,
    "epochs":         2,          # 2 epochs max — avoid overfit
    "save_steps":     500,
    "eval_steps":     200,
    "log_steps":      50,
    "target_langs":   ["English", "Hindi", "Telugu"],
    "min_response_len": 10,       # filter very short responses
    "max_prompt_len": 512,        # cap prompt length
}

# ─── LOAD TOKENIZER ────────────────────────────────────────────────────────────

log.info("Loading tokenizer...")
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOK_PATH,
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
    unk_token="<|unk|>",
)
# Add chat template
fast_tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
log.info(f"Tokenizer loaded: {fast_tokenizer.vocab_size} vocab")

# ─── LOAD BASE MODEL ───────────────────────────────────────────────────────────

log.info(f"Loading pretrained model from {BASE_MODEL_PATH}...")
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.config.use_cache = False  # disable for training
log.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# ─── LOAD & PREPARE SFT DATASETS ─────────────────────────────────────────────

log.info("\nLoading Aya instruction dataset...")

LANGUAGE_FILTERS = {
    "English": ["en", "eng", "english"],
    "Hindi":   ["hi", "hin", "hindi"],
    "Telugu":  ["te", "tel", "telugu"],
}

def load_aya_filtered():
    """Load Aya dataset filtered to our 3 target languages"""
    try:
        aya = load_dataset("CohereForAI/aya_dataset", split="train")
        log.info(f"Aya dataset loaded: {len(aya):,} total examples")

        # Filter to our languages
        def is_target_lang(example):
            lang = example.get("language", "").lower()
            for target, aliases in LANGUAGE_FILTERS.items():
                if any(a in lang for a in aliases):
                    return True
            return False

        filtered = aya.filter(is_target_lang, num_proc=4)
        log.info(f"After language filter: {len(filtered):,} examples")
        return filtered

    except Exception as e:
        log.error(f"Failed to load Aya: {e}")
        log.info("Trying alternative: Aya Collection")
        try:
            aya = load_dataset("CohereForAI/aya_collection_language_split",
                              split="train", streaming=False)
            return aya
        except Exception as e2:
            log.error(f"Aya Collection also failed: {e2}")
            return None

aya_data = load_aya_filtered()

# Also add FLAN subset for English reasoning
log.info("Loading small FLAN subset for English reasoning...")
try:
    flan = load_dataset("Muennighoff/flan", split="train",
                        streaming=False).select(range(50000))
    log.info(f"FLAN subset: {len(flan):,} examples")
except Exception as e:
    log.warning(f"FLAN not available: {e}")
    flan = None

# ─── FORMAT CONVERSATIONS ─────────────────────────────────────────────────────

def format_aya_to_chat(example):
    """Convert Aya format to chat template format"""
    inputs   = example.get("inputs", example.get("prompt", "")).strip()
    targets  = example.get("targets", example.get("completion", "")).strip()

    if not inputs or not targets:
        return None
    if len(targets.split()) < SFT_CFG["min_response_len"]:
        return None

    messages = [
        {"role": "user",      "content": inputs},
        {"role": "assistant", "content": targets},
    ]
    text = fast_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

def format_flan_to_chat(example):
    """Convert FLAN format to chat format"""
    inputs  = example.get("inputs", "").strip()
    targets = example.get("targets", "").strip()
    if not inputs or not targets or len(targets.split()) < 5:
        return None
    messages = [
        {"role": "user",      "content": inputs},
        {"role": "assistant", "content": targets},
    ]
    text = fast_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

log.info("Formatting datasets to chat template...")
formatted_datasets = []

if aya_data:
    aya_formatted = []
    for ex in aya_data:
        formatted = format_aya_to_chat(ex)
        if formatted:
            aya_formatted.append(formatted)
    log.info(f"Aya formatted: {len(aya_formatted):,} examples")
    formatted_datasets.extend(aya_formatted)

if flan:
    flan_formatted = []
    for ex in flan:
        formatted = format_flan_to_chat(ex)
        if formatted:
            flan_formatted.append(formatted)
    log.info(f"FLAN formatted: {len(flan_formatted):,} examples")
    formatted_datasets.extend(flan_formatted)

log.info(f"Total SFT examples: {len(formatted_datasets):,}")

# Distribution check
sample_texts = [d["text"] for d in formatted_datasets[:1000]]
lang_counts = {"Hindi": 0, "Telugu": 0, "English": 0}
for t in sample_texts:
    if any('\u0900' <= c <= '\u097F' for c in t):
        lang_counts["Hindi"] += 1
    elif any('\u0C00' <= c <= '\u0C7F' for c in t):
        lang_counts["Telugu"] += 1
    else:
        lang_counts["English"] += 1
log.info(f"Language distribution in sample: {lang_counts}")

if lang_counts["Telugu"] == 0:
    log.warning("⚠ No Telugu examples found in SFT data — check Aya dataset language codes")

# Convert to HF Dataset
from datasets import Dataset as HFDataset
sft_dataset = HFDataset.from_list(formatted_datasets)

# Split 95/5 train/val
split = sft_dataset.train_test_split(test_size=0.05, seed=42)
train_sft = split["train"]
val_sft   = split["test"]
log.info(f"SFT train: {len(train_sft):,}, val: {len(val_sft):,}")

# ─── TRAINING ─────────────────────────────────────────────────────────────────

log.info("\nStarting SFT training...")

sft_args = SFTConfig(
    output_dir=SFT_CKPT_DIR,
    num_train_epochs=SFT_CFG["epochs"],
    per_device_train_batch_size=SFT_CFG["batch_size"],
    per_device_eval_batch_size=SFT_CFG["batch_size"],
    gradient_accumulation_steps=SFT_CFG["grad_accum"],
    learning_rate=SFT_CFG["peak_lr"],
    lr_scheduler_type="cosine",
    warmup_steps=SFT_CFG["warmup_steps"],
    weight_decay=SFT_CFG["weight_decay"],
    max_grad_norm=SFT_CFG["grad_clip"],
    bf16=True,
    tf32=False,
    logging_steps=SFT_CFG["log_steps"],
    save_steps=SFT_CFG["save_steps"],
    eval_steps=SFT_CFG["eval_steps"],
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",   # set to "wandb" if you have wandb configured
    dataloader_num_workers=4,
    remove_unused_columns=True,
    # SFT-specific
    max_seq_length=SFT_CFG["max_seq_len"],
    dataset_text_field="text",
    packing=True,        # pack multiple short examples per sequence
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_sft,
    eval_dataset=val_sft,
    tokenizer=fast_tokenizer,
)

log.info("Running SFT...")
trainer.train()

log.info("Saving final SFT model...")
final_path = f"{SFT_CKPT_DIR}/final"
trainer.save_model(final_path)
fast_tokenizer.save_pretrained(final_path)

log.info(f"""
╔══════════════════════════════════════════════════════════════════╗
║                   SFT COMPLETE                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Model saved: {final_path}
║                                                                  ║
║  Next steps:                                                     ║
║  1. python3 scripts/05_eval.py     — run benchmarks             ║
║  2. Push to HuggingFace:                                         ║
║     huggingface-cli upload ncncomplete/KarnaLM-360M-chat \\    ║
║       {final_path}                                               ║
╚══════════════════════════════════════════════════════════════════╝
""")
