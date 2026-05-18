import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": [],
        "metadata": {
            "kaggle": {
                "accelerator": "none",
                "isGpuEnabled": False,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    for ctype, src in cells:
        nb["cells"].append({
            "cell_type": ctype,
            "metadata": {},
            "source": [line + "\n" for line in src.split('\n')[:-1]] + [src.split('\n')[-1]] if src else []
        })
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def get_shared_cells(lang_name, lang_code, target_tokens, config_dict, sources_json, language_filters):
    cell_1 = """# ─── CELL 1: Install & Setup ───
import subprocess, sys

def install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(args))

print("Installing packages...")
install("datasets==2.20.0", "huggingface_hub", "datasketch", "pandas", "pyarrow")

from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, login

try:
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=hf_token, add_to_git_credential=False)
    api = HfApi(token=hf_token)
    print("✓ HF Login Success")
except Exception as e:
    print(f"⚠ HF_TOKEN not found in Kaggle Secrets.")
    print("DEBUG: Make sure you added 'HF_TOKEN' in Add-ons -> Secrets and CHECKED the box to attach it.")
    print(f"Error: {e}")
    
    from getpass import getpass
    hf_token = getpass("Enter HF Token manually to continue: ")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        api = HfApi(token=hf_token)
        print("✓ HF Login Success (Manual)")
    else:
        print("⚠ Proceeding without HF authentication (some steps will fail).")
        api = None
    
import gc
import json
import time
import os
import re
import hashlib
from itertools import islice
import pandas as pd
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
"""

    cell_2 = f"""# ─── CELL 2: CONFIG ───
HF_CHECKPOINT_REPO = "ncncomplete/karnalm-checkpoints"
HF_DATA_REPO = "ncncomplete/karnalm-data"
LANGUAGE = "{lang_name}"
LANGUAGE_CODE = "{lang_code}"

if api:
    try:
        api.create_repo(HF_CHECKPOINT_REPO, repo_type="dataset", exist_ok=True, private=True)
        api.create_repo(HF_DATA_REPO, repo_type="dataset", exist_ok=True, private=True)
    except Exception as e:
        print(f"⚠ Could not create repos: {{e}}")

CONFIG = {json.dumps(config_dict, indent=2)}

SOURCES = {sources_json}
"""

    cell_3 = f"""# ─── CELL 3: Helper Functions ───
def strip_html(text):
    return re.sub(r'<[^>]+>', '', text)

def check_repetition(text):
    lines = text.split('\\n')
    line_counts = {{}}
    for line in lines:
        line = line.strip()
        if not line: continue
        line_counts[line] = line_counts.get(line, 0) + 1
        if line_counts[line] > 5:
            return True
    return False

{language_filters}

def is_valid_doc(text, ex, source_config):
    if not text: return False
    
    clean_text = strip_html(text)
    if len(clean_text) < CONFIG["min_doc_length"] or len(clean_text) > CONFIG["max_doc_length"]:
        return False
        
    if check_repetition(clean_text): 
        return False
        
    if not passes_language_filters(clean_text, ex, source_config):
        return False
    
    return True

exact_hashes = set()

def build_fresh_lsh():
    return MinHashLSH(threshold=CONFIG["minhash_threshold"], num_perm=CONFIG["minhash_num_perm"])

lsh = build_fresh_lsh()

def compute_minhash(text):
    m = MinHash(num_perm=CONFIG["minhash_num_perm"])
    t = text.lower()
    for j in range(max(1, len(t)-4)):
        m.update(t[j:j+5].encode('utf-8'))
    return m

def get_exact_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
"""

    cell_4 = """# ─── CELL 4: Checkpoint System ───
class CheckpointManager:
    def __init__(self):
        self.filename = f"{LANGUAGE_CODE}_checkpoint.json"
        self.local_path = f"/kaggle/working/{self.filename}"
        self.state = {
            "completed_sources": [],
            "current_source": None,
            "docs_processed_in_current": 0,
            "total_unique_docs": 0,
            "total_chars": 0,
            "output_shards_pushed": []
        }
        self.load_checkpoint()
        
    def load_checkpoint(self):
        if not api: return
        try:
            path = hf_hub_download(repo_id=HF_CHECKPOINT_REPO, repo_type="dataset", filename=self.filename, token=hf_token)
            with open(path, 'r', encoding='utf-8') as f:
                self.state = json.load(f)
            print(f"✓ Resumed checkpoint from HF: {self.filename}")
            # Note: HF keeps full file history, so pruning old checkpoints is handled by HF natively.
        except Exception as e:
            print("No existing checkpoint found on HF. Starting fresh.")
            
    def save_checkpoint(self):
        if not api: return
        with open(self.local_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)
        try:
            api.upload_file(
                path_or_fileobj=self.local_path,
                path_in_repo=self.filename,
                repo_id=HF_CHECKPOINT_REPO,
                repo_type="dataset",
                commit_message=f"Checkpoint update (Docs: {self.state['total_unique_docs']})"
            )
        except Exception as e:
            print(f"  ⚠ Failed to push checkpoint: {e}")

    def push_shard(self, source_key, is_partial=False):
        if not api: return
        buffer_file = f"/kaggle/working/{LANGUAGE_CODE}_{source_key}_buffer.jsonl"
        if not os.path.exists(buffer_file):
            return
            
        try:
            df = pd.read_json(buffer_file, lines=True)
            if df.empty:
                os.remove(buffer_file)
                return
                
            parquet_name = f"partial_{source_key}.parquet" if is_partial else f"{source_key}.parquet"
            path_in_repo = f"{LANGUAGE_CODE}/{parquet_name}"
            parquet_path = buffer_file.replace('.jsonl', '.parquet')
            
            df.to_parquet(parquet_path, index=False)
            
            api.upload_file(
                path_or_fileobj=parquet_path,
                path_in_repo=path_in_repo,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                commit_message=f"Push shard {path_in_repo}"
            )
            print(f"  ✓ Shard {path_in_repo} pushed to HF")
            
            if path_in_repo not in self.state["output_shards_pushed"]:
                self.state["output_shards_pushed"].append(path_in_repo)
                
            # Cleanup local files to save space
            os.remove(buffer_file)
            os.remove(parquet_path)
        except Exception as e:
            print(f"  ⚠ Failed to push shard for {source_key}: {e}")
"""

    cell_5 = """# ─── CELL 5: Main Pipeline Loop ───
ckpt = CheckpointManager()
start_time = time.time()
lsh_doc_count = 0
skipped_filter = 0
skipped_dup = 0
out_f = None
global lsh

try:
    for source in SOURCES:
        s_key = source["key"]
        
        if s_key in ckpt.state["completed_sources"]:
            print(f"⏭️ Skipping completed source: {s_key}")
            continue
            
        print(f"\\n━━ Processing Source: {s_key} ━━")
        
        if ckpt.state["current_source"] != s_key:
            ckpt.state["current_source"] = s_key
            ckpt.state["docs_processed_in_current"] = 0
            ckpt.save_checkpoint()
            
        buffer_file = f"/kaggle/working/{LANGUAGE_CODE}_{s_key}_buffer.jsonl"
        
        # Restore buffer if resuming in an ephemeral container
        if ckpt.state["docs_processed_in_current"] > 0 and not os.path.exists(buffer_file):
            partial_parquet = f"partial_{s_key}.parquet"
            try:
                print(f"  Downloading existing partial shard {partial_parquet} to restore local buffer...")
                if api:
                    path = hf_hub_download(repo_id=HF_DATA_REPO, repo_type="dataset", filename=f"{LANGUAGE_CODE}/{partial_parquet}", token=hf_token)
                    df = pd.read_parquet(path)
                    df.to_json(buffer_file, orient='records', lines=True, force_ascii=False)
                    print("  ✓ Local buffer restored from HF partial shard.")
            except Exception as e:
                print(f"  ⚠ Could not download partial shard (starting fresh buffer): {e}")

        # Always open in append mode. If we resume, we just append more records to the buffer.
        out_f = open(buffer_file, 'a', encoding='utf-8')
        
        try:
            print(f"  Loading dataset: {source['dataset']}")
            kwargs = {"streaming": True, "split": source.get("split", "train")}
            if "config" in source: kwargs["name"] = source["config"]
            if source.get("trust_remote_code"): kwargs["trust_remote_code"] = True
            
            ds = load_dataset(source["dataset"], **kwargs)
            
            skip_n = ckpt.state["docs_processed_in_current"]
            if skip_n > 0:
                print(f"  Resuming {s_key} from doc {skip_n}...")
                ds_iter = islice(iter(ds), skip_n, None)
            else:
                ds_iter = iter(ds)
                
            limit = source.get("limit", None)
            
            chunk = []
            pbar = tqdm(desc=f"  {s_key}", unit="doc")
            
            for ex in ds_iter:
                if limit and (ckpt.state["docs_processed_in_current"] >= limit):
                    print(f"\\n  Reached configured limit of {limit} for {s_key}.")
                    break
                if ckpt.state["total_unique_docs"] >= CONFIG["target_unique_docs"]:
                    print(f"\\n  🎯 Target reached!") # finally block will handle pushing the partial shard for this source
                    break
                    
                # Note: this counts raw stream position, not unique valid docs
                ckpt.state["docs_processed_in_current"] += 1
                pbar.update(1)
                
                # Custom text extraction if needed (e.g. Wikipedia title + text)
                if source.get("text_concat"):
                    fields = source["text_concat"]
                    text = "\\n".join([str(ex.get(f, "")) for f in fields])
                else:
                    text = ex.get(source.get("text_field", "text"), "")
                    
                if not isinstance(text, str): text = str(text)
                text = text.strip()
                
                if not is_valid_doc(text, ex, source):
                    skipped_filter += 1
                    continue
                    
                # Exact Dedup
                h_exact = get_exact_hash(text)
                if h_exact in exact_hashes:
                    skipped_dup += 1
                    continue
                    
                chunk.append((text, h_exact))
                
                # Chunk Processing
                if len(chunk) >= CONFIG["stream_chunk_size"]:
                    for doc_text, doc_hash in chunk:
                        if ckpt.state["total_unique_docs"] >= CONFIG["target_unique_docs"]: break
                        
                        m = compute_minhash(doc_text)
                        if len(lsh.query(m)) > 0:
                            skipped_dup += 1
                            continue
                            
                        lsh.insert(f"d_{ckpt.state['total_unique_docs']}", m)
                        exact_hashes.add(doc_hash)
                        
                        doc_obj = {"text": doc_text, "source": s_key, "lang": LANGUAGE_CODE}
                        out_f.write(json.dumps(doc_obj, ensure_ascii=False) + '\\n')
                        
                        ckpt.state["total_unique_docs"] += 1
                        ckpt.state["total_chars"] += len(doc_text)
                        lsh_doc_count += 1
                        
                        if lsh_doc_count >= CONFIG["lsh_rebuild_every"]:
                            lsh = build_fresh_lsh()
                            lsh_doc_count = 0
                            
                    chunk = []
                    out_f.flush()
                    
                    pbar.set_postfix({
                        "unique": f"{ckpt.state['total_unique_docs']:,}",
                        "dups": f"{skipped_dup:,}"
                    })
                    
                # Checkpoint Progress
                if ckpt.state["docs_processed_in_current"] % CONFIG["CHECKPOINT_PUSH_EVERY"] == 0:
                    out_f.flush()
                    ckpt.save_checkpoint()
                    elapsed = time.time() - start_time
                    est_tokens = ckpt.state['total_chars'] / 4.0
                    print(f"\\n  [Checkpoint] {s_key} | Processed: {ckpt.state['docs_processed_in_current']:,} | "
                          f"Unique: {ckpt.state['total_unique_docs']:,} | Tokens: {est_tokens/1e9:.3f}B | "
                          f"Time: {elapsed/60:.1f}m")
            
            # Flush remaining chunk
            for doc_text, doc_hash in chunk:
                if ckpt.state["total_unique_docs"] >= CONFIG["target_unique_docs"]: break
                m = compute_minhash(doc_text)
                if len(lsh.query(m)) > 0:
                    skipped_dup += 1
                    continue
                lsh.insert(f"d_{ckpt.state['total_unique_docs']}", m)
                exact_hashes.add(doc_hash)
                doc_obj = {"text": doc_text, "source": s_key, "lang": LANGUAGE_CODE}
                out_f.write(json.dumps(doc_obj, ensure_ascii=False) + '\\n')
                ckpt.state["total_unique_docs"] += 1
                ckpt.state["total_chars"] += len(doc_text)
                lsh_doc_count += 1
                
            chunk = []
            out_f.flush()
            pbar.close()
            
            # Source fully complete
            out_f.close()
            if ckpt.state["total_unique_docs"] < CONFIG["target_unique_docs"]:
                print(f"✓ Source completed: {s_key}")
                ckpt.push_shard(s_key, is_partial=False)
                ckpt.state["completed_sources"].append(s_key)
                ckpt.state["current_source"] = None
                ckpt.state["docs_processed_in_current"] = 0
                ckpt.save_checkpoint()
            else:
                break
                
        except Exception as e:
            print(f"  ✗ Failed processing {s_key}: {e}")
            if out_f and not out_f.closed:
                out_f.close()
            continue

except KeyboardInterrupt:
    print("\\n🛑 Interrupted by user.")
except Exception as e:
    print(f"\\n💥 Unexpected pipeline error: {e}")
finally:
    print("\\n--- Pipeline Run Ended ---")
    if getattr(ckpt, 'state', {}).get("current_source"):
        print(f"Pushing partial buffer and checkpoint before exiting...")
        current = ckpt.state["current_source"]
        try:
            if out_f is not None and not out_f.closed:
                out_f.close()
        except Exception:
            pass
        ckpt.push_shard(current, is_partial=True)
        ckpt.save_checkpoint()
    print("Checkpoint saved. Resume by re-running this notebook.")
"""

    cell_6 = """# ─── CELL 6: Final Summary ───
print("="*50)
print("📊 RUN SUMMARY")
print(f"Total Unique Docs: {ckpt.state['total_unique_docs']:,}")
print(f"Total Characters:  {ckpt.state['total_chars']:,}")
print(f"Estimated Tokens:  {ckpt.state['total_chars'] / 4.0 / 1e9:.3f} Billion")
print(f"Completed Sources: {ckpt.state['completed_sources']}")
print(f"Output Shards:     {len(ckpt.state['output_shards_pushed'])}")
print("="*50)
"""
    return [
        ("code", cell_1),
        ("code", cell_2),
        ("code", cell_3),
        ("code", cell_4),
        ("code", cell_5),
        ("code", cell_6)
    ]


# ---- HINDI NOTEBOOK ----
hi_config = {
    "min_doc_length": 100,
    "max_doc_length": 50000,
    "script_ratio": 0.30,
    "minhash_threshold": 0.85,
    "minhash_num_perm": 64,
    "target_unique_docs": 6000000,
    "stream_chunk_size": 10000,
    "lsh_rebuild_every": 300000,
    "CHECKPOINT_PUSH_EVERY": 100000
}
hi_sources = [
    {"key": "sangraha_verified", "dataset": "ai4bharat/sangraha", "config": "verified/hin", "split": "train", "trust_remote_code": True},
    {"key": "sangraha_unverified", "dataset": "ai4bharat/sangraha", "config": "unverified/hin", "split": "train", "trust_remote_code": True},
    {"key": "sangraha_synthetic", "dataset": "ai4bharat/sangraha", "config": "synthetic/hin", "split": "train", "trust_remote_code": True},
    {"key": "indiccorp_hi", "dataset": "ai4bharat/IndicCorp", "config": "hi", "split": "train", "trust_remote_code": True},
    {"key": "mc4_hi", "dataset": "mc4", "config": "hi", "split": "train", "extra_filter": "hi_60"}
]
hi_filters = """def passes_language_filters(text, ex, source_config):
    alpha_chars = sum(1 for c in text if c.isalpha())
    script_chars = sum(1 for c in text if '\\u0900' <= c <= '\\u097F')
    
    if alpha_chars > 0 and (script_chars / alpha_chars) < CONFIG["script_ratio"]:
        return False
        
    if source_config.get("extra_filter") == "hi_60":
        if len(text) > 0 and (script_chars / len(text)) < 0.60:
            return False
            
    return True"""
create_notebook("hindi_pipeline.ipynb", get_shared_cells("Hindi", "hi", "~5.5B tokens", hi_config, json.dumps(hi_sources, indent=4), hi_filters))

# ---- TELUGU NOTEBOOK ----
te_config = {
    "min_doc_length": 100,
    "max_doc_length": 50000,
    "script_ratio": 0.30,
    "minhash_threshold": 0.85,
    "minhash_num_perm": 64,
    "target_unique_docs": 5000000,
    "stream_chunk_size": 10000,
    "lsh_rebuild_every": 300000,
    "CHECKPOINT_PUSH_EVERY": 100000
}
te_sources = [
    {"key": "sangraha_verified", "dataset": "ai4bharat/sangraha", "config": "verified/tel", "split": "train", "trust_remote_code": True},
    {"key": "sangraha_unverified", "dataset": "ai4bharat/sangraha", "config": "unverified/tel", "split": "train", "trust_remote_code": True},
    {"key": "indiccorp_te", "dataset": "ai4bharat/IndicCorp", "config": "te", "split": "train", "trust_remote_code": True},
    {"key": "oscar_te", "dataset": "oscar-corpus/OSCAR-2301", "config": "te", "split": "train", "text_field": "content", "extra_filter": "te_60"},
    {"key": "wikipedia_te", "dataset": "wikimedia/wikipedia", "config": "20231101.te", "split": "train", "text_concat": ["title", "text"]}
]
te_filters = """def passes_language_filters(text, ex, source_config):
    alpha_chars = sum(1 for c in text if c.isalpha())
    script_chars = sum(1 for c in text if '\\u0C00' <= c <= '\\u0C7F')
    
    if alpha_chars > 0 and (script_chars / alpha_chars) < CONFIG["script_ratio"]:
        return False
        
    if source_config.get("extra_filter") == "te_60":
        if len(text) > 0 and (script_chars / len(text)) < 0.60:
            return False
            
    return True"""
create_notebook("telugu_pipeline.ipynb", get_shared_cells("Telugu", "te", "~3B tokens", te_config, json.dumps(te_sources, indent=4), te_filters))

# ---- ENGLISH NOTEBOOK ----
en_config = {
    "min_doc_length": 150,
    "max_doc_length": 100000,
    "minhash_threshold": 0.85,
    "minhash_num_perm": 64,
    "target_unique_docs": 15000000,
    "stream_chunk_size": 10000,
    "lsh_rebuild_every": 300000,
    "CHECKPOINT_PUSH_EVERY": 100000
}
en_sources = [
    {"key": "fineweb_main", "dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "split": "train", "limit": 8000000},
    {"key": "fineweb_edu", "dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train", "limit": 5000000, "extra_filter": "en_edu"},
    {"key": "openwebtext", "dataset": "Skylion007/openwebtext", "split": "train"}
]
en_filters = """def passes_language_filters(text, ex, source_config):
    words = text.split()
    if not words: return False
    
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length < 3 or avg_word_length > 15:
        return False
        
    non_space_chars = sum(1 for c in text if not c.isspace())
    alpha_chars = sum(1 for c in text if c.isalpha())
    if non_space_chars > 0 and (alpha_chars / non_space_chars) < 0.70:
        return False
        
    # Reject cross-script noise
    for c in text:
        if '\\u0900' <= c <= '\\u097F' or '\\u0C00' <= c <= '\\u0C7F' or '\\u4E00' <= c <= '\\u9FFF':
            return False
            
    if source_config.get("extra_filter") == "en_edu":
        score = ex.get("score") or ex.get("educational_score")
        if score is None or float(score) < 2.0:
            return False
        
    return True
"""
create_notebook("english_pipeline.ipynb", get_shared_cells("English", "en", "~13B tokens", en_config, json.dumps(en_sources, indent=4), en_filters))
