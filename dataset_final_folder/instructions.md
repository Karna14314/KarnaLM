
# Multilingual Pretraining Data Preprocessing Pipeline

This document outlines the architecture and configuration for the 3 Kaggle notebooks (Python `.ipynb` files) handling multilingual pretraining data preprocessing. Each notebook is self-contained, handles one language, and is designed to survive Kaggle's 12-hour session limit by using HuggingFace dataset-based checkpointing.

## === SHARED ARCHITECTURE FOR ALL 3 NOTEBOOKS ===

### CHECKPOINT SYSTEM:
- **Location:** Checkpoint is stored as a JSON file pushed to HuggingFace dataset repo: `ncncomplete/karnalm-checkpoints`
- **Naming Convention:** `{language}_checkpoint.json`
- **Startup Logic:** On every notebook start, pull checkpoint from HF if it exists, else start fresh.
- **Checkpoint Structure:**
  ```json
  {
    "completed_sources": [],           // list of fully done source keys
    "current_source": "source_key",    // source currently being processed
    "docs_processed_in_current": 0,    // how many docs done in current source
    "total_unique_docs": 0,
    "total_tokens_estimated": 0,
    "output_shards_pushed": []         // list of shard filenames pushed to HF
  }
  ```
- **Push Frequency:** Push checkpoint to HF every 25,000 docs processed.
- **Source Completion:** When a source fully completes: add to `completed_sources`, push its output shard to HF dataset repo `ncncomplete/karnalm-data` as `{language}_{source_key}.jsonl`, then push updated checkpoint.

### OUTPUT SYSTEM:
- Each source writes to its own JSONL shard.
- Shard pushed to HuggingFace dataset: `ncncomplete/karnalm-data`
- Local buffer file: `/kaggle/working/{language}_{source_key}_buffer.jsonl`
- Each doc in output JSONL: `{"text": "...", "source": "source_key", "lang": "xx"}`

### DEDUPLICATION:
- Use MinHash LSH (`datasketch` library) for fuzzy dedup.
- Parameters: `num_perm=128`, `threshold=0.85`
- LSH index is rebuilt fresh each session (not checkpointed — acceptable).
- Exact dedup via sha256 hash set maintained in memory.

### RESUME LOGIC on notebook start:
1. Pull checkpoint JSON from HF.
2. Skip all sources in `completed_sources` entirely.
3. For `current_source`: stream the HF dataset, skip first `docs_processed_in_current` docs, then continue processing.
4. For sources after `current_source`: process fresh.

### TEXT FILTERING (apply to all languages):
- Min length: 100 characters.
- Max length: 100,000 characters.
- Remove docs where >30% characters are non-language-script.
- Remove docs with excessive repetition (any line repeated >5 times).
- Strip HTML tags before length check.

### PROGRESS DISPLAY:
- Print progress every 10,000 docs: source name, docs processed, unique kept, estimated tokens, elapsed time.
- Print checkpoint push confirmations.
- Print shard push confirmations.

### HF AUTHENTICATION:
- Use `HfApi` with token from kaggle secrets key `HF_TOKEN`.
- Use `kaggle_secrets`: `UserSecretsClient().get_secret("HF_TOKEN")`

## === NOTEBOOK 1: HINDI (hindi_pipeline.ipynb) ===
**Language code:** `hi`
**Script:** Devanagari — Unicode range `\u0900-\u097F`
**Target:** ~5.5B tokens

**Sources in processing order:**
1. **source_key:** `sangraha_verified`
   - dataset: `ai4bharat/sangraha`, config: `verified/hin`, split: `train`
2. **source_key:** `sangraha_unverified`
   - dataset: `ai4bharat/sangraha`, config: `unverified/hin`, split: `train`
3. **source_key:** `sangraha_synthetic`
   - dataset: `ai4bharat/sangraha`, config: `synthetic/hin`, split: `train`
4. **source_key:** `indiccorp_hi`
   - dataset: `ai4bharat/IndicCorp`, config: `hi`, split: `train`, text field: `text`
5. **source_key:** `mc4_hi`
   - dataset: `mc4`, config: `hi`, split: `train`, text field: `text`
   - **Extra filter:** reject docs where Hindi script chars < 60% of total chars.

## === NOTEBOOK 2: TELUGU (telugu_pipeline.ipynb) ===
**Language code:** `te`
**Script:** Telugu — Unicode range `\u0C00-\u0C7F`
**Target:** ~3B tokens

**Sources in processing order:**
1. **source_key:** `sangraha_verified`
   - dataset: `ai4bharat/sangraha`, config: `verified/tel`, split: `train`
2. **source_key:** `sangraha_unverified`
   - dataset: `ai4bharat/sangraha`, config: `unverified/tel`, split: `train`
3. **source_key:** `indiccorp_te`
   - dataset: `ai4bharat/IndicCorp`, config: `te`, split: `train`, text field: `text`
4. **source_key:** `oscar_te`
   - dataset: `oscar-corpus/OSCAR-2301`, config: `te`, split: `train`, text field: `content`
   - **Extra filter:** reject docs where Telugu script chars < 60% of total chars.
5. **source_key:** `wikipedia_te`
   - dataset: `wikimedia/wikipedia`, config: `20231101.te`, split: `train`, text field: `text`

## === NOTEBOOK 3: ENGLISH (english_pipeline.ipynb) ===
**Language code:** `en`
**Script:** Latin — no strict script filter needed
**Target:** ~13B tokens

**Sources in processing order:**
1. **source_key:** `fineweb_1`
   - dataset: `HuggingFaceFW/fineweb`, config: `sample-10BT`, split: `train`, text field: `text`
   - Stream only first 3,000,000 docs then mark complete.
2. **source_key:** `fineweb_2`
   - dataset: `HuggingFaceFW/fineweb`, config: `sample-10BT`, split: `train`, text field: `text`
   - Skip first 3,000,000 docs, stream next 3,000,000 then mark complete. (Offset implemented in source config).
3. **source_key:** `fineweb_edu`
   - dataset: `HuggingFaceFW/fineweb-edu`, config: `sample-10BT`, split: `train`, text field: `text`
   - Stream first 2,000,000 docs then mark complete.

**Extra English filters:**
- Reject docs with avg word length < 3 or > 15.
- Reject docs where alphabetic chars < 70% of non-space chars.

## === ADDITIONAL REQUIREMENTS ===
- All notebooks must install dependencies in cell 1: `pip install datasets huggingface_hub datasketch kaggle-secrets`
- Use `load_dataset(..., streaming=True)` for all sources — never download full datasets.
- The fast-forward skip for resume must use `itertools.islice`, not a loop with a counter, for memory efficiency.
- At notebook end (or keyboard interrupt via try/finally): push whatever is in the local buffer + push current checkpoint before dying.
- Add a CONFIG cell near the top where only these values appear (easy to edit): `HF_CHECKPOINT_REPO`, `HF_DATA_REPO`, `CHECKPOINT_PUSH_EVERY`, `LANGUAGE`, `LANGUAGE_CODE`, `SOURCES` list.
- Token estimation: use `(total_chars / 4)` as a rough estimate.
