# Telugu Data Pipeline for KarnaLM (2.5-3B Tokens)

## Files Created

| File | Description |
|------|-------------|
| `notebook_telugu_pipeline.ipynb` | Kaggle notebook (recommended) |
| `telugu_pipeline.py` | Standalone Python script |

## Target
- **5M+ unique Telugu documents**
- **~2.5-3 billion tokens** (at 500 tokens/document average)
- **Sources**: Sangraha Telugu, IndicCorp v2 Telugu, OSCAR Telugu, Wikipedia Telugu

## Method
Uses the same streaming single-pass approach as the Hindi pipeline:
- Stream → filter → dedup → write (never loads all docs into RAM)
- MinHashLSH deduplication with periodic index rebuilds
- Telugu script detection: >30% of alphabetic chars must be Telugu (U+0C00–U+0C7F)

---

## Option 1: Run on Kaggle (Recommended)

### Setup
1. Go to [Kaggle](https://www.kaggle.com) → Notebooks → "New Notebook"
2. Settings → Accelerator: **None (CPU)**
3. Add Secret: `HF_TOKEN` (from HuggingFace)

### Run
1. Upload `notebook_telugu_pipeline.ipynb` to Kaggle
2. Run all cells
3. Runtime: ~6-8 hours
4. Output: `/kaggle/working/telugu_filtered.jsonl`

### Upload to HF
If Cell 10 (upload) times out:
1. Go to **Output** tab → download `telugu_filtered.jsonl`
2. Upload manually to HuggingFace dataset

---

## Option 2: Run Locally

### Requirements
```bash
pip install datasets==2.20.0 tokenizers sentencepiece huggingface_hub datasketch tqdm
```

### Set Token
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Run
```bash
python telugu_pipeline.py
```

---

## Configuration (Both)

Key settings in `CONFIG` dict:

```python
"target_unique_docs": 5_000_000,   # Target 5M unique docs (~2.5-3B tokens)
"telugu_script_ratio": 0.30,          # >30% Telugu chars required
"minhash_threshold": 0.85,           # Deduplication threshold
"stream_chunk_size": 10_000,        # Process 10K docs at a time
"lsh_rebuild_every": 300_000,      # Rebuild index every 300K docs
```

---

## Output Format

JSONL file with one document per line:
```json
{"text": "తెలుగు వచనం ఇక్కడ ఉంది..."}
{"text": "మరో డాక్యుమెంట్..."}
```

---

## Comparison: Hindi vs Telugu Pipelines

| Feature | Hindi | Telugu (New) |
|---------|-------|--------------|
| Target Docs | 2M | 5M |
| Est. Tokens | ~1-1.5B | ~2.5-3B |
| Script Range | U+0900–U+097F | U+0C00–U+0C7F |
| Sources | Sangraha, IndicCorp | Sangraha, IndicCorp, OSCAR, Wiki |
| Memory | Streaming | Streaming |
