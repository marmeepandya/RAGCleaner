#!/usr/bin/env python
# coding: utf-8

# # Experiment 7 — PyDI-Integrated RAG Pipeline (TE+RR)
# 
# This notebook re-runs the best configuration (TE+RR) but with PyDI modules replacing
# manual code wherever possible.
# 
# ## What changed vs exp_runner Exp5
# 
# | Step | Exp5 (manual) | Exp7 (PyDI-integrated) |
# |------|--------------|------------------------|
# | KB preparation | Raw load | **PyDI `NormalizationSpec` + `transform_dataframe`** |
# | Candidate formatting | Raw 5 rows to LLM | **PyDI `DataFusionEngine` fuses top-5 → 1 clean record** |
# | Evaluation | Manual CE + exact match | Manual CE + exact match + **PyDI `DataFusionEvaluator`** |
# | Retrieval | Manual cosine (unchanged) | Manual cosine (unchanged — PyDI blocker needs live embedder) |
# | Reranking | Manual CrossEncoder (unchanged) | Manual CrossEncoder (unchanged) |
# 
# ## Hypothesis
# Fusing the top-5 retrieved candidates into a single clean record before prompting the LLM
# should reduce noise and field confusion errors — potentially improving over Exp5's 83.3% CE eval.
# 
# ## Prerequisites
# - `exp_setup.ipynb` must have been run
# - `embeddings/openai_kb.pt` and `embeddings/openai_query.pt` must exist
# - `results_mit_UNKNOWN/exp5_rag_openai_reranker.csv` must exist for comparison

# ## 1. Environment Setup

# In[ ]:


import sys
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib64/python3.12/site-packages')
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib/python3.12/site-packages')
sys.path.append('/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI')

import os, re, glob, time, random, threading, math
import torch
import numpy as np
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder, util

# ── PyDI imports ──────────────────────────────────────────────────────────────
from PyDI.normalization import NormalizationSpec, transform_dataframe
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    DataFusionEvaluator,
    voting,
    median,
    longest_string,
    tokenized_match,
    numeric_tolerance_match,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
print('PyDI imports OK')


# ## 2. Configuration
# Identical to exp_runner — same attributes, same retrieval parameters.

# In[ ]:


TARGET_ATTRIBUTES  = ['bus_type', 'model_number', 'model',
                      'read_speed_mb_s', 'write_speed_mb_s', 'height_mm', 'width_mm']
NUMERIC_ATTRIBUTES = {'read_speed_mb_s', 'write_speed_mb_s', 'height_mm', 'width_mm'}
TEXT_ATTRIBUTES    = {'bus_type', 'model_number', 'model'}

HF_CACHE       = '/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub'
EMBEDDINGS_DIR = 'embeddings'
RESULTS_DIR    = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_N = 20   # Stage 1: cosine retrieval
TOP_K = 5    # Stage 2: after CrossEncoder reranking

SKIP_FIELDS = {'id', 'url', 'description', 'title_description',
               'price', 'priceCurrency', 'cluster_id'}

print(f'TOP_N={TOP_N} | TOP_K={TOP_K}')


# ## 3. Load Datasets and Eval Set

# In[ ]:


DATA_DIR = 'normalized_products'
df1      = pd.read_json(f'{DATA_DIR}/dataset_1_normalized.json')
df2      = pd.read_json(f'{DATA_DIR}/dataset_2_normalized.json')
df3      = pd.read_json(f'{DATA_DIR}/dataset_3_normalized.json')
df4      = pd.read_json(f'{DATA_DIR}/dataset_4_normalized.json')

assert os.path.exists('eval_set.csv'),      'Run exp_setup.ipynb first'
assert os.path.exists('query_indices.csv'), 'Run exp_setup.ipynb first'

eval_df       = pd.read_csv('eval_set.csv')
query_indices = pd.read_csv('query_indices.csv').iloc[:, 0].tolist()
query_df      = df1.loc[query_indices].copy()
query_idx_to_pos = {idx: pos for pos, idx in enumerate(query_df.index)}

print(f'KB rows: {len(df2)+len(df3)+len(df4):,} | Query: {len(query_df)} | Eval tasks: {len(eval_df)}')
print('Tasks per attribute:')
print(eval_df['attribute'].value_counts().to_string())


# ## 4. PyDI Step 1 — Normalize KB with PyDI
# 
# Before building the KB we apply `PyDI.normalization` to clean values:
# - Strip whitespace from all target attributes
# - Cast numeric attributes to float (avoids string '3480.0' vs float 3480.0 mismatches)
# - Preserve original casing for text (unlike Exp6 — we need readable values for the LLM prompt)

# In[ ]:


def normalize_dataset(df, dataset_name):
    """Apply PyDI normalization — strip whitespace, cast numerics, preserve text casing."""
    df = df.copy()
    df.attrs['dataset_name'] = dataset_name

    spec = NormalizationSpec()

    # Text attributes: strip whitespace only — preserve casing for LLM readability
    for attr in TEXT_ATTRIBUTES:
        if attr in df.columns:
            spec.set_column(attr,
                            output_type='string',
                            strip_whitespace=True,
                            on_failure='keep')

    # Numeric attributes: cast to float for clean median fusion
    for attr in NUMERIC_ATTRIBUTES:
        if attr in df.columns:
            spec.set_column(attr,
                            output_type='float',
                            strip_whitespace=True,
                            on_failure='null')

    result = transform_dataframe(df, spec)
    normalized = result.dataframe
    normalized.attrs['dataset_name'] = dataset_name
    print(f'  {dataset_name}: {result.total_transformed} normalized, {result.total_failed} failures')
    return normalized


print('Normalizing datasets with PyDI...')
df2_norm = normalize_dataset(df2, 'dataset_2')
df3_norm = normalize_dataset(df3, 'dataset_3')
df4_norm = normalize_dataset(df4, 'dataset_4')

# Build the KB from normalized datasets
kb = pd.concat([df2_norm, df3_norm, df4_norm], ignore_index=True)
print(f'\nNormalized KB: {len(kb):,} rows')


# ## 5. PyDI Step 2 — Define Fusion Strategy for Candidate Consolidation
# 
# After retrieval we have top-5 KB candidates. Instead of passing all 5 raw rows to the LLM,
# we use PyDI's `DataFusionStrategy` to consolidate them into a single clean record.
# 
# This reduces noise and field confusion — the LLM sees one authoritative value per attribute
# rather than 5 potentially conflicting ones.

# In[ ]:


# Fusion strategy for consolidating retrieved candidates
candidate_strategy = DataFusionStrategy('candidate_fusion_strategy')

# Text: voting — most frequent value among top-5 candidates
candidate_strategy.add_attribute_fuser('bus_type',         voting)
candidate_strategy.add_attribute_fuser('model',            voting)
# model_number: longest_string — most specific / complete SKU
candidate_strategy.add_attribute_fuser('model_number',     longest_string)
# Numeric: median — robust to outlier candidates
candidate_strategy.add_attribute_fuser('read_speed_mb_s',  median)
candidate_strategy.add_attribute_fuser('write_speed_mb_s', median)
candidate_strategy.add_attribute_fuser('height_mm',        median)
candidate_strategy.add_attribute_fuser('width_mm',         median)

# Evaluation functions — same tolerances as is_correct_standard
candidate_strategy.add_evaluation_function('bus_type',         tokenized_match, threshold=0.8)
candidate_strategy.add_evaluation_function('model',            tokenized_match, threshold=0.8)
candidate_strategy.add_evaluation_function('model_number',     tokenized_match, threshold=0.9)
candidate_strategy.add_evaluation_function('read_speed_mb_s',  numeric_tolerance_match, tolerance=0.10)
candidate_strategy.add_evaluation_function('write_speed_mb_s', numeric_tolerance_match, tolerance=0.10)
candidate_strategy.add_evaluation_function('height_mm',        numeric_tolerance_match, tolerance=0.10)
candidate_strategy.add_evaluation_function('width_mm',         numeric_tolerance_match, tolerance=0.10)

# Fusion engine — no debug file needed per-task, we run it inline
candidate_engine = DataFusionEngine(candidate_strategy, debug=False)

print('Candidate fusion strategy ready.')
print('Resolvers: voting (bus_type, model) | longest_string (model_number) | median (numeric)')


# ## 6. Load Pre-computed OpenAI Embeddings and CrossEncoder

# In[ ]:


oai_kb_embs    = torch.load(f'{EMBEDDINGS_DIR}/openai_kb.pt')
oai_query_embs = torch.load(f'{EMBEDDINGS_DIR}/openai_query.pt')
print(f'OpenAI embeddings — KB: {oai_kb_embs.shape} | Query: {oai_query_embs.shape}')

CE_SNAP = glob.glob(f'{HF_CACHE}/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/*/')
CE_PATH = CE_SNAP[0].rstrip('/') if CE_SNAP else 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder = CrossEncoder(CE_PATH)
print(f'CrossEncoder loaded from: {CE_PATH}')


# ## 7. Evaluation and Helper Functions
# Identical to exp_runner — copied verbatim.

# In[ ]:


def is_correct_standard(predicted, ground_truth, attribute):
    if not predicted or str(predicted).strip().lower() in {'', 'nan', 'none', 'unknown', 'null'}:
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(',', '').strip())
            g = float(str(ground_truth).replace(',', '').strip())
            return abs(p - g) / abs(g) <= 0.10 if g != 0 else p == 0
        except:
            pass
    p, g = str(predicted).lower().strip(), str(ground_truth).lower().strip()
    return p == g or p in g or g in p


def evaluate_ce(predicted, ground_truth, attribute):
    if predicted == 'UNKNOWN' or str(predicted).lower() in {'nan', 'none', 'null', ''}:
        return 'wrong'
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(',', '').strip())
            g = float(str(ground_truth).replace(',', '').strip())
            if g == 0: return 'correct' if p == 0 else 'wrong'
            r = abs(p - g) / abs(g)
            return 'correct' if r <= 0.10 else ('acceptable' if r <= 0.30 else 'wrong')
        except:
            return 'wrong'
    score = cross_encoder.predict([[ground_truth, predicted]])[0]
    return 'correct' if score > 2.0 else ('acceptable' if score > -1.0 else 'wrong')


def parse_response(text, attribute):
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith('VALUE:'):
            val = line.split(':', 1)[1].strip().strip('"').strip("'")
            return 'UNKNOWN' if val.upper() in {'UNKNOWN', 'NONE', 'NAN', 'NULL', ''} else val
    pat = rf'{attribute}\s*[:\u2192>\-]+\s*([^\s|]+)'
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip('"').strip("'").strip('[]')
        if val.upper() != 'UNKNOWN' and len(val) > 2 and val.lower() not in {'none', 'nan', 'null', 'exact', 'value'}:
            return val
    if attribute in NUMERIC_ATTRIBUTES:
        nums = re.findall(r'\b\d+\.?\d*\b', text)
        if nums: return nums[0]
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.upper().startswith('VALUE:'): cleaned = cleaned.split(':', 1)[1].strip()
    if cleaned and len(cleaned) < 80 and '\n' not in cleaned and \
       cleaned.upper() not in {'UNKNOWN', 'NONE', 'NULL', 'NAN', ''}:
        return cleaned
    return 'UNKNOWN'


def fix_prediction(pred):
    if isinstance(pred, str) and pred.strip().upper().startswith('VALUE:'):
        val = pred.strip().split(':', 1)[1].strip()
        return 'UNKNOWN' if val.upper() in {'UNKNOWN', 'NONE', 'NULL', 'NAN', ''} else val
    return pred


def retrieve_top_n(q_emb, kb_embs, n):
    scores  = util.cos_sim(q_emb, kb_embs)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:n]
    return top_idx


def rerank(query_text, top_n_idx, kb_texts, k):
    cands_texts = [kb_texts[i] for i in top_n_idx]
    pairs       = [[query_text[:300], t[:300]] for t in cands_texts]
    scores      = cross_encoder.predict(pairs)
    top_k_local = np.argsort(-scores)[:k]
    return top_n_idx[top_k_local], scores[top_k_local]


def predict_with_timeout(predict_fn, timeout=300):
    result = ['UNKNOWN']
    def target():
        try: result[0] = predict_fn()
        except: result[0] = 'UNKNOWN'
    t = threading.Thread(target=target)
    t.start(); t.join(timeout=timeout)
    if t.is_alive(): print('  ⚠️ TIMEOUT'); return 'UNKNOWN'
    return result[0]


def evaluate_and_save(results_df, config_name, filename):
    results_df['predicted']        = results_df['predicted'].apply(fix_prediction)
    results_df['unknown']          = results_df['predicted'] == 'UNKNOWN'
    results_df['correct_standard'] = results_df.apply(
        lambda r: is_correct_standard(r['predicted'], r['ground_truth'], r['attribute']), axis=1)
    results_df['ce_judgment'] = [
        evaluate_ce(r['predicted'], r['ground_truth'], r['attribute'])
        for _, r in results_df.iterrows()]
    path = f'{RESULTS_DIR}/{filename}'
    results_df.to_csv(path, index=False)

    std = results_df['correct_standard'].mean()
    ce  = results_df['ce_judgment'].isin(['correct', 'acceptable']).mean()
    unk = results_df['unknown'].mean()
    print(f'\n{"="*60}\nRESULTS — {config_name}\n{"="*60}')
    print(f'Standard accuracy:    {std:.3f} ({std*100:.1f}%)')
    print(f'CE eval (c+a):        {ce:.3f} ({ce*100:.1f}%)')
    print(f'UNKNOWN rate:         {unk:.3f} ({unk*100:.1f}%)')
    print(f'Total tasks:          {len(results_df)}')
    print('\nPer-attribute:')
    print(results_df.groupby('attribute').agg(
        n=('correct_standard', 'count'),
        std_acc=('correct_standard', 'mean'),
        ce_acc=('ce_judgment', lambda x: x.isin(['correct', 'acceptable']).mean()),
        unknown=('unknown', 'mean')
    ).round(3).to_string())
    print(f'\n✓ Saved to {path}')
    return results_df

print('All helper functions ready.')


# ## 8. PyDI Candidate Fusion Helper
# 
# This is the core new function. Given top-K retrieved candidates (a DataFrame of KB rows),
# it uses PyDI's `DataFusionEngine` to fuse them into a single consolidated record.
# 
# The fused record is then formatted for the LLM prompt — cleaner than passing 5 raw rows.

# In[ ]:


def fuse_candidates_with_pydi(candidates):
    """
    Use PyDI DataFusionEngine to consolidate top-K retrieved candidates
    into a single fused record.

    Returns a dict of {attribute: fused_value} for all target attributes.
    Falls back to the first candidate's value if fusion fails.
    """
    if len(candidates) == 0:
        return {}

    # Add required _id and dataset_name for PyDI engine
    cands = candidates.reset_index(drop=True).copy()
    cands['_id'] = cands.index
    cands.attrs['dataset_name'] = 'retrieved_candidates'

    # Build correspondences: all candidates describe the same entity (the query product)
    # Score = 1.0 — these are our retrieved matches, we treat them as confirmed
    corr_rows = []
    ids = cands['_id'].tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            corr_rows.append({'id1': ids[i], 'id2': ids[j], 'score': 1.0})
    if len(ids) == 1:
        corr_rows.append({'id1': ids[0], 'id2': ids[0], 'score': 1.0})

    correspondences = pd.DataFrame(corr_rows)

    try:
        fused = candidate_engine.run(
            datasets=[cands],
            correspondences=correspondences,
            id_column='_id',
            include_singletons=True,
        )
        if len(fused) == 0:
            return {}
        # Take first (and only) fused record
        row = fused.iloc[0]
        result = {}
        for attr in TARGET_ATTRIBUTES:
            val = row.get(attr, None)
            if val is not None and pd.notna(val):
                result[attr] = val
        return result
    except Exception as e:
        # Fallback: return first candidate's values
        print(f'    [fusion fallback: {e}]')
        row = cands.iloc[0]
        return {attr: row.get(attr) for attr in TARGET_ATTRIBUTES
                if pd.notna(row.get(attr, None))}


def format_fused_candidate(fused_values, candidates):
    """
    Format the PyDI-fused record for the LLM prompt.
    Shows: fused attribute values + supporting info from the best candidate.
    """
    lines = []

    # Line 1: fused attribute values (the PyDI output)
    fused_fields = {k: str(v) for k, v in fused_values.items()
                    if str(v).strip().lower() not in {'', 'nan', 'none'}}
    if fused_fields:
        lines.append('  [PyDI fused] ' + ' | '.join(f'{k}: {v}' for k, v in fused_fields.items()))

    # Lines 2+: individual candidates for context (title + brand only)
    for _, c in candidates.iterrows():
        title = c.get('title', c.get('title_description', ''))
        brand = c.get('brand', '')
        if pd.notna(title) and str(title).strip():
            line = f'  - title: {str(title)[:120]}'
            if pd.notna(brand) and str(brand).strip():
                line += f' | brand: {brand}'
            lines.append(line)

    return '\n'.join(lines) if lines else '  (no candidates)'


print('PyDI candidate fusion functions ready.')


# ## 9. LLM Prompts
# 
# Same prompts as exp_runner but adapted for the fused candidate format.
# The LLM now sees:
# 1. A single `[PyDI fused]` line with the consolidated attribute values
# 2. Individual candidate titles for context
# 
# This is cleaner than 5 raw rows with all fields.

# In[ ]:


FEW_SHOT_PROMPTS_PYDI = {

'bus_type': """\
You are a product data expert filling missing values in a product database.
The reference below shows a PyDI-fused consensus value from multiple KB sources, plus individual product titles.
Use the fused value as your primary source. Only deviate if the titles clearly contradict it.
Step 1: Check the [PyDI fused] bus_type value.
Step 2: Confirm against the product titles. Copy the value exactly.
Respond with VALUE:<answer> only. If uncertain, respond with VALUE:UNKNOWN.

Example:
Query: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache - WD60EZAZ
Reference:
  [PyDI fused] bus_type: SATA III | model_number: WD60EZAZ
  - title: WD Blue 6TB Hard Drive WD60EZAZ | brand: Western Digital
VALUE:SATA III

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:""",

'model_number': """\
You are a product data expert filling missing values in a product database.
The reference shows a PyDI-fused model_number plus individual product titles.
Copy the EXACT model_number character by character.
CRITICAL: model_numbers like GV-N166SOC-6GD and GV-N1660OC-6GD are DIFFERENT.
If the fused value conflicts with titles or you are uncertain, respond with VALUE:UNKNOWN.

Example:
Query: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4 NVMe
Reference:
  [PyDI fused] model_number: CSSD-F960GBMP510 | bus_type: PCIe 3.0 x4
  - title: Corsair Force MP510 960GB NVMe SSD | brand: Corsair
VALUE:CSSD-F960GBMP510

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:""",

'model': """\
You are a product data expert filling missing values in a product database.
The reference shows a PyDI-fused model name plus individual product titles.
Copy the model value from the fused reference exactly.
Respond with VALUE:<answer> only. If uncertain, respond with VALUE:UNKNOWN.

Example:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe
Reference:
  [PyDI fused] model: Force Series MP510 | model_number: CSSD-F960GBMP510
  - title: Corsair Force MP510 960GB NVMe SSD | brand: Corsair
VALUE:Force Series MP510

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:""",

'read_speed_mb_s': """\
You are a product data expert filling missing values in a product database.
The reference shows a PyDI-fused read_speed_mb_s (median of retrieved candidates).
Return a number only. Do NOT return the write speed.
Respond with VALUE:<number> only.

Example:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference:
  [PyDI fused] read_speed_mb_s: 3480.0 | write_speed_mb_s: 3000.0
  - title: Corsair Force MP510 960GB NVMe | brand: Corsair
VALUE:3480

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:""",

'write_speed_mb_s': """\
You are a product data expert filling missing values in a product database.
The reference shows a PyDI-fused write_speed_mb_s (median of retrieved candidates).
Return a number only. Do NOT return the read speed.
Respond with VALUE:<number> only.

Example:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference:
  [PyDI fused] read_speed_mb_s: 3480.0 | write_speed_mb_s: 3000.0
  - title: Corsair Force MP510 960GB NVMe | brand: Corsair
VALUE:3000

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:""",

'height_mm': """\
You are a product data expert filling missing values in a product database.
The reference shows a PyDI-fused height_mm (median of retrieved candidates).
Return a number only. Do NOT confuse height with width or length.
Respond with VALUE:<number> only.

Example:
Query: MSI GeForce GTX 1660 Ti GAMING X 6G graphics card
Reference:
  [PyDI fused] height_mm: 46.0 | width_mm: 127.0
  - title: MSI GTX 1660 Ti GAMING X 6G | brand: MSI
VALUE:46

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:""",

'width_mm': """\
You are a product data expert filling missing values in a product database.
The reference shows a PyDI-fused width_mm (median of retrieved candidates).
Return a number only. Do NOT confuse width with height or length.
Respond with VALUE:<number> only.

Example:
Query: MSI GeForce GTX 1660 Ti GAMING X 6G graphics card
Reference:
  [PyDI fused] height_mm: 46.0 | width_mm: 127.0
  - title: MSI GTX 1660 Ti GAMING X 6G | brand: MSI
VALUE:127

Now fill the missing value:
Query: {text}
Reference:
{candidates}
VALUE:"""
}

print(f'PyDI-adapted prompts ready for: {list(FEW_SHOT_PROMPTS_PYDI.keys())}')


# ## 10. LLM Setup

# In[ ]:


predict_model = ChatOllama(
    model='llama3.1:8b',
    temperature=0,
    seed=42,
    base_url='http://127.0.0.1:11435'
)
print(f'Ollama: {repr(predict_model.invoke("Say OK").content[:20])}')


# ## 11. Experiment 7 — TE+RR with PyDI Candidate Fusion
# 
# Pipeline per task:
# 1. Embed query (OpenAI pre-computed)
# 2. Retrieve top-20 by cosine similarity
# 3. CrossEncoder rerank → top-5
# 4. **PyDI normalize + fuse top-5 → 1 clean record** ← new
# 5. LLM extracts from fused record
# 6. Evaluate with existing functions

# In[ ]:


EXP7_FILE = 'exp7_pydi_te_reranker.csv'
EXP7_PATH = f'{RESULTS_DIR}/{EXP7_FILE}'

# KB texts for CrossEncoder reranking (same as exp_runner)
def row_to_text(row):
    td = row.get('title_description', '')
    if pd.notna(td) and str(td).strip(): return str(td).strip()[:400]
    attrs = ['title', 'model', 'model_number', 'brand', 'product_type']
    text  = ' | '.join([str(row[a]) for a in attrs if pd.notna(row.get(a))])
    desc  = row.get('description', '')
    if pd.notna(desc) and str(desc).strip(): text += ' | ' + str(desc).strip()[:200]
    return text

kb_texts_oai = kb.apply(row_to_text, axis=1).tolist()

if os.path.exists(EXP7_PATH):
    print('Loading existing Exp 7 results...')
    exp7_df = pd.read_csv(EXP7_PATH)
else:
    print('Running Exp 7 — TE+RR with PyDI Candidate Fusion...')
    t0, predictions = time.time(), []

    for i, (_, task) in enumerate(eval_df.iterrows()):
        idx, attr, gt = task['df1_idx'], task['attribute'], task['ground_truth']
        text       = query_df.loc[idx, 'title_description']
        query_text = row_to_text(query_df.loc[idx])
        q_emb      = oai_query_embs[query_idx_to_pos[idx]]

        # Stage 1: cosine retrieval
        top_n_idx = retrieve_top_n(q_emb, oai_kb_embs, TOP_N)

        # Stage 2: CrossEncoder rerank
        top_k_idx, _ = rerank(query_text, top_n_idx, kb_texts_oai, TOP_K)
        candidates   = kb.iloc[top_k_idx]

        # Stage 3 (NEW): PyDI normalize + fuse candidates → 1 clean record
        fused_values = fuse_candidates_with_pydi(candidates)
        formatted    = format_fused_candidate(fused_values, candidates)

        # Stage 4: LLM extracts from fused record
        def predict():
            prompt = FEW_SHOT_PROMPTS_PYDI[attr].format(
                text=str(text)[:500],
                candidates=formatted
            )
            return parse_response(
                predict_model.invoke([HumanMessage(content=prompt)]).content.strip(),
                attr
            )

        predicted = predict_with_timeout(predict)
        eta = (time.time() - t0) / (i + 1) * (len(eval_df) - i - 1)
        print(f'  [{i+1}/{len(eval_df)}] Row {idx} | {attr:<22} | '
              f'GT: {str(gt):<25} | Pred: {predicted:<25} | ETA: {eta/60:.1f}min')

        predictions.append({
            'df1_idx':     idx,
            'config':      'PyDI-TE-Reranker',
            'attribute':   attr,
            'is_numeric':  task['is_numeric'],
            'ground_truth': gt,
            'predicted':   predicted,
            'unknown':     predicted == 'UNKNOWN',
            'fused_value': fused_values.get(attr, 'N/A'),
        })

    exp7_df = evaluate_and_save(pd.DataFrame(predictions), 'Exp 7: PyDI TE+RR', EXP7_FILE)


# ## 12. Full Comparison — Exp5 vs Exp7
# Direct apples-to-apples: same retrieval (TE+RR), different candidate processing.

# In[ ]:


all_files = {
    'Exp1: LLM-only':            'results_mit_UNKNOWN/exp1_llm_only.csv',
    'Exp2: RAG-MiniLM':          'results_mit_UNKNOWN/exp2_rag_minilm.csv',
    'Exp3: MiniLM+Reranker':     'results_mit_UNKNOWN/exp3_rag_minilm_reranker.csv',
    'Exp4: BGE+Reranker':        'results_mit_UNKNOWN/exp4_rag_bge_reranker.csv',
    'Exp5: TE+RR (manual)':      'results_mit_UNKNOWN/exp5_rag_openai_reranker.csv',
    'Exp6: PyDI KB Fusion':      'results/exp6_pydi_fusion.csv',
    'Exp7: TE+RR + PyDI Fusion': f'{RESULTS_DIR}/{EXP7_FILE}',
}

print(f'{"Configuration":<35} {"Std acc":>10} {"CE eval":>10} {"UNKNOWN":>10} {"n":>6}')
print('-' * 75)

all_dfs = {}
for name, path in all_files.items():
    if not os.path.exists(path):
        print(f'  {name:<35} (not found)')
        continue
    df = pd.read_csv(path)
    all_dfs[name] = df
    std = df['correct_standard'].mean()
    ce  = df['ce_judgment'].isin(['correct', 'acceptable']).mean()
    unk = df['unknown'].mean()
    marker = ' ← PyDI' if 'PyDI' in name else ''
    print(f'  {name:<35} {std:>10.3f} {ce:>10.3f} {unk:>10.3f} {len(df):>6}{marker}')

# Per-attribute comparison Exp5 vs Exp7
exp5_path = 'results_mit_UNKNOWN/exp5_rag_openai_reranker.csv'
if os.path.exists(exp5_path) and os.path.exists(EXP7_PATH):
    print('\nPer-attribute: Exp5 (manual TE+RR) vs Exp7 (PyDI TE+RR)')
    e5 = pd.read_csv(exp5_path)
    e7 = pd.read_csv(EXP7_PATH)
    comp = pd.DataFrame({
        'Exp5_std':    e5.groupby('attribute')['correct_standard'].mean(),
        'Exp7_std':    e7.groupby('attribute')['correct_standard'].mean(),
        'Exp5_ce':     e5.groupby('attribute')['ce_judgment'].apply(lambda x: x.isin(['correct','acceptable']).mean()),
        'Exp7_ce':     e7.groupby('attribute')['ce_judgment'].apply(lambda x: x.isin(['correct','acceptable']).mean()),
    }).round(3)
    comp['delta_std'] = (comp['Exp7_std'] - comp['Exp5_std']).round(3)
    comp['delta_ce']  = (comp['Exp7_ce']  - comp['Exp5_ce']).round(3)
    print(comp.to_string())

