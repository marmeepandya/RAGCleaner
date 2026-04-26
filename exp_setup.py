#!/usr/bin/env python
# coding: utf-8

# # RAG-Driven Data Cleaning — Experiment Setup
# 
# ## Purpose of this Notebook
# 
# This notebook is the **single entry point** for all experiments. It handles:
# 
# 1. **Evaluation set construction** — builds the canonical eval set (10 instances per attribute) and saves it to disk. All subsequent experiments load this exact same set.
# 2. **LLM-only baseline (Exp 1)** — Llama 3.1 8B predicts missing values from product text alone with no KB access.
# 3. **Embedding generation and persistence** — encodes the KB and query rows using multiple embedding models and saves dense vectors to disk. Retrieval experiments load pre-computed embeddings rather than recomputing each time.
# 
# ## Why Pre-compute and Save Embeddings?
# 
# Encoding 2,200 KB rows with a large model like BGE-large takes several minutes on an A100 GPU. Since the KB does not change between experiments, recomputing embeddings on every run is wasteful. Saving tensors to `.pt` files means all retrieval experiments can be run repeatedly without the encoding overhead.
# 
# ## Embedding Models Compared
# 
# | Model | Dimensions | Parameters | Notes |
# |---|---|---|---|
# | `all-MiniLM-L6-v2` | 384 | 22M | Fast, lightweight baseline retriever |
# | `BAAI/bge-large-en-v1.5` | 1024 | 335M | Top MTEB retrieval model, requires query prefix |
# | `intfloat/e5-large-v2` | 1024 | 335M | Strong retrieval model, requires query/passage prefix |
# | `dunzhang/stella_en_1.5B_v5` | 1024 | 1.5B | State-of-art on MTEB, very large |
# | `text-embedding-3-large` | 3072 | API | OpenAI model, strongest but costs money |
# 
# Dense vector embeddings represent text as fixed-size floating point arrays in high-dimensional space. Semantic similarity is computed as cosine similarity between these vectors — products describing similar hardware will have embeddings pointing in similar directions regardless of surface-level word differences.
# 
# ## Determinism
# 
# - `random.seed(42)` and `np.random.seed(42)` for Python/NumPy
# - `temperature=0` and `seed=42` for Ollama inference
# - Eval set saved to CSV once — all experiments reload the same rows
# - Embeddings saved to `.pt` files — same float tensors reused across all runs
# - BGE snapshot path pinned explicitly rather than using glob

# ## 1. Environment Setup

# In[1]:


import sys
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib64/python3.12/site-packages')
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib/python3.12/site-packages')
sys.path.append('/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI')

import os, re, glob, time, random, threading
import torch
import numpy as np
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer, CrossEncoder, util

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"}')
if torch.cuda.is_available():
    print(f'VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ## 2. Configuration

# In[2]:


TARGET_ATTRIBUTES = ['bus_type','model_number','model','read_speed_mb_s','write_speed_mb_s','height_mm','width_mm']
NUMERIC_ATTRIBUTES = {'read_speed_mb_s','write_speed_mb_s','height_mm','width_mm'}
TEXT_ATTRIBUTES    = {'bus_type','model_number','model'}

MIN_PER_ATTR   = 10
MAX_QUERY_ROWS = 50

DATA_DIR       = 'normalized_products'
EMBEDDINGS_DIR = 'embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

EVAL_SET_FILE      = 'eval_set.csv'
QUERY_INDICES_FILE = 'query_indices.csv'
LLM_ONLY_RESULT    = 'exp1_llm_only.csv'
HF_CACHE           = '/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub'

print(f'MIN_PER_ATTR: {MIN_PER_ATTR} | MAX_QUERY_ROWS: {MAX_QUERY_ROWS}')
print(f'Embeddings will be saved to: {EMBEDDINGS_DIR}/')


# ## 3. Load Datasets

# In[3]:


df1     = pd.read_json(f'{DATA_DIR}/dataset_1_normalized.json')
df2     = pd.read_json(f'{DATA_DIR}/dataset_2_normalized.json')
df3     = pd.read_json(f'{DATA_DIR}/dataset_3_normalized.json')
df4     = pd.read_json(f'{DATA_DIR}/dataset_4_normalized.json')
kb_full = pd.concat([df2, df3, df4], ignore_index=True)
kb      = kb_full.copy()

print(f'Dataset 1 (query pool): {len(df1):,} rows')
print(f'KB combined:            {len(kb):,} rows')


# ## 4. Build and Save Evaluation Set
# 
# **Ground truth**: first non-null value for the target attribute from any KB row sharing the same `cluster_id` as the query product.
# 
# **Selection**: greedy — rows added until every attribute has `MIN_PER_ATTR` instances.
# 
# **Persistence**: saved to CSV so all experiments use the exact same tasks. This is critical for fair comparison — without this, each experiment could accidentally evaluate on slightly different rows due to ordering non-determinism.

# In[4]:


def get_ground_truth(cluster_id, attribute):
    matches = kb_full[kb_full['cluster_id'] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {'','none','nan'}:
            return str(val).strip()
    return None


if os.path.exists(EVAL_SET_FILE):
    print(f'Loading existing eval set...')
    eval_df       = pd.read_csv(EVAL_SET_FILE)
    query_indices = pd.read_csv(QUERY_INDICES_FILE).iloc[:,0].tolist()
    query_df      = df1.loc[query_indices].copy()
else:
    print('Building eval set...')
    all_candidates = []
    for idx, row in df1.iterrows():
        missing = []
        for attr in TARGET_ATTRIBUTES:
            if pd.isna(row.get(attr)):
                gt = get_ground_truth(row['cluster_id'], attr)
                if gt: missing.append((attr, gt))
        if missing:
            all_candidates.append({'df1_idx': idx, 'missing_attrs': missing})

    attr_counts, selected_rows, selected_idx = {a:0 for a in TARGET_ATTRIBUTES}, [], set()
    for cand in all_candidates:
        if len(selected_rows) >= MAX_QUERY_ROWS: break
        if any(attr_counts[a] < MIN_PER_ATTR for a,_ in cand['missing_attrs']):
            selected_rows.append(cand)
            selected_idx.add(cand['df1_idx'])
            for a,_ in cand['missing_attrs']: attr_counts[a] += 1
    for cand in all_candidates:
        if len(selected_rows) >= MAX_QUERY_ROWS: break
        if cand['df1_idx'] not in selected_idx:
            selected_rows.append(cand)
            selected_idx.add(cand['df1_idx'])
            for a,_ in cand['missing_attrs']: attr_counts[a] += 1

    records = [{'df1_idx':item['df1_idx'],'attribute':a,'ground_truth':gt,'is_numeric':a in NUMERIC_ATTRIBUTES}
               for item in selected_rows for a,gt in item['missing_attrs']]
    eval_df       = pd.DataFrame(records)
    query_indices = [item['df1_idx'] for item in selected_rows]
    query_df      = df1.loc[query_indices].copy()
    eval_df.to_csv(EVAL_SET_FILE, index=False)
    pd.Series(query_indices).to_csv(QUERY_INDICES_FILE, index=False, header=['df1_idx'])
    print(f'Saved to {EVAL_SET_FILE}')

print(f'\nQuery rows: {len(query_df)} | Eval tasks: {len(eval_df)}')
print('\nTasks per attribute:')
for a in TARGET_ATTRIBUTES:
    n = (eval_df['attribute']==a).sum()
    print(f'  {a:<25} {n:>3}  {"✓" if n>=MIN_PER_ATTR else "✗ NEED MORE"}')


# ## 5. Sanity Check — Ground Truth Present in KB
# 
# Every task must have its ground truth value present verbatim in the KB. Tasks where the value is missing from the KB are excluded — it would be unfair to penalise the model for a KB completeness issue rather than a retrieval or extraction failure.

# In[5]:


print('=== SANITY CHECK: Ground truth present in KB ===\n')
found = not_found = 0
for _, task in eval_df.iterrows():
    idx, attr, gt = task['df1_idx'], task['attribute'], task['ground_truth']
    cluster_id    = query_df.loc[idx,'cluster_id']
    kb_matches    = kb[kb['cluster_id']==cluster_id]
    ok = any(pd.notna(r.get(attr)) and str(r.get(attr)).strip()==str(gt).strip() for _,r in kb_matches.iterrows())
    found += int(ok); not_found += int(not ok)
    print(f'{"✓" if ok else "✗"} Row {idx:<4} | {attr:<22} | GT: {str(gt):<30} | KB matches: {len(kb_matches)}')
print(f'\nCoverage: {found}/{found+not_found} = {100*found/(found+not_found):.1f}%')


# ## 6. Evaluation Functions
# 
# ### Standard Accuracy
# Substring matching with ±10% numeric tolerance. Lenient — used as a quick sanity check.
# 
# ### CrossEncoder (CE) Evaluation
# Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score semantic similarity between prediction and ground truth. Removes self-evaluation bias that would occur if the same LLM judged its own outputs. Thresholds calibrated empirically on this dataset: score > 2.0 = correct, > -1.0 = acceptable.

# In[6]:


CE_SNAP = glob.glob(f'{HF_CACHE}/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/*/')
CE_PATH = CE_SNAP[0].rstrip('/') if CE_SNAP else 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder = CrossEncoder(CE_PATH)
print(f'CrossEncoder: {CE_PATH}')

def is_correct_standard(predicted, ground_truth, attribute):
    if not predicted or str(predicted).strip().lower() in {'','nan','none','unknown','null'}:
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(',','').strip())
            g = float(str(ground_truth).replace(',','').strip())
            return abs(p-g)/abs(g) <= 0.10 if g != 0 else p == 0
        except: pass
    p, g = str(predicted).lower().strip(), str(ground_truth).lower().strip()
    return p == g or p in g or g in p

def evaluate_ce(predicted, ground_truth, attribute):
    if predicted == 'UNKNOWN' or str(predicted).lower() in {'nan','none','null',''}:
        return 'wrong'
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(',','').strip())
            g = float(str(ground_truth).replace(',','').strip())
            if g == 0: return 'correct' if p == 0 else 'wrong'
            r = abs(p-g)/abs(g)
            return 'correct' if r<=0.10 else ('acceptable' if r<=0.30 else 'wrong')
        except: return 'wrong'
    score = cross_encoder.predict([[ground_truth, predicted]])[0]
    return 'correct' if score > 2.0 else ('acceptable' if score > -1.0 else 'wrong')

def fix_prediction(pred):
    if isinstance(pred, str) and pred.strip().upper().startswith('VALUE:'):
        val = pred.strip().split(':',1)[1].strip()
        return 'UNKNOWN' if val.upper() in {'UNKNOWN','NONE','NULL','NAN',''} else val
    return pred

def print_summary(df, name):
    std = df['correct_standard'].mean()
    ce  = df['ce_judgment'].isin(['correct','acceptable']).mean()
    unk = df['unknown'].mean()
    print(f'\n{"="*55}\nRESULTS — {name}\n{"="*55}')
    print(f'Standard accuracy:    {std:.3f} ({std*100:.1f}%)')
    print(f'CE eval (c+a):        {ce:.3f} ({ce*100:.1f}%)')
    print(f'UNKNOWN rate:         {unk:.3f} ({unk*100:.1f}%)')
    print(f'Total tasks:          {len(df)}')
    print('\nPer-attribute:')
    print(df.groupby('attribute').agg(
        n=('correct_standard','count'),
        std_acc=('correct_standard','mean'),
        ce_acc=('ce_judgment', lambda x: x.isin(['correct','acceptable']).mean()),
        unknown=('unknown','mean')
    ).round(3).to_string())

print('Evaluation functions ready.')


# ## 7. LLM Setup and Prompts (Experiment 1 — LLM-only)

# In[8]:


predict_model = ChatOllama(model='llama3.1:8b', temperature=0, seed=42, base_url='http://127.0.0.1:11435')
print(f'Ollama: {repr(predict_model.invoke("Say OK").content[:20])}')

FEW_SHOT_PROMPTS_LLM = {
'bus_type': """\
You are a product data expert. Extract ONLY the bus_type from the product text.
bus_type is the interface/connection type (e.g. PCIe 3.0 x16, SATA III, USB 3.0).
Respond with VALUE:<answer> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache - WD60EZAZ
VALUE:SATA III

Example 2:
Product: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4, NVMe, up to 3480/3000MB/s
VALUE:PCIe 3.0 x4

Now extract:
Product: {text}
VALUE:""",

'model_number': """\
You are a product data expert. Extract ONLY the model_number from the product text.
model_number is the exact manufacturer SKU — alphanumeric with dashes, often at the end of the title.
Respond with VALUE:<answer> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: WD Blue 6TB Desktop Hard Disk Drive - SATA 6Gb/s 256MB Cache 3.5 Inch - WD60EZAZ
VALUE:WD60EZAZ

Example 2:
Product: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4 (CSSD-F960GBMP510)
VALUE:CSSD-F960GBMP510

Now extract:
Product: {text}
VALUE:""",

'model': """\
You are a product data expert. Extract ONLY the model name from the product text.
model is the product model name — not the full title, not the SKU.
Respond with VALUE:<answer> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache - WD60EZAZ
VALUE:WD Blue

Example 2:
Product: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4, NVMe, up to 3480/3000MB/s
VALUE:Force Series MP510

Now extract:
Product: {text}
VALUE:""",

'read_speed_mb_s': """\
You are a product data expert. Extract ONLY the sequential READ speed in MB/s.
Return the number only. Do NOT return write speed.
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: CORSAIR Force MP510 960GB M.2 SSD, up to 3480/3000MB/s read/write
VALUE:3480

Example 2:
Product: Samsung 860 EVO 4TB. Description: Read 550MB/s, Write 520MB/s
VALUE:550

Now extract:
Product: {text}
VALUE:""",

'write_speed_mb_s': """\
You are a product data expert. Extract ONLY the sequential WRITE speed in MB/s.
Return the number only. Do NOT return read speed.
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: CORSAIR Force MP510 960GB M.2 SSD, up to 3480/3000MB/s read/write
VALUE:3000

Example 2:
Product: Samsung 860 EVO 4TB. Description: Read 550MB/s, Write 520MB/s
VALUE:520

Now extract:
Product: {text}
VALUE:""",

'height_mm': """\
You are a product data expert. Extract ONLY the HEIGHT in millimeters.
Return the number only. Do NOT confuse with width or length.
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: GeForce GTX 1660 Ti. Description: Dimensions: 46mm x 127mm x 247mm (H x W x D)
VALUE:46

Example 2:
Product: Samsung T5 1TB SSD. Description: Dimensions (HDW): 10.5mm x 74mm x 57mm
VALUE:10.5

Now extract:
Product: {text}
VALUE:""",

'width_mm': """\
You are a product data expert. Extract ONLY the WIDTH in millimeters.
Return the number only. Do NOT confuse with height or length.
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: GeForce GTX 1660 Ti. Description: Dimensions: 46mm x 127mm x 247mm (H x W x D)
VALUE:127

Example 2:
Product: Samsung T5 1TB SSD. Description: Dimensions (HDW): 10.5mm x 74mm x 57mm
VALUE:57

Now extract:
Product: {text}
VALUE:"""
}
print(f'Prompts ready for: {list(FEW_SHOT_PROMPTS_LLM.keys())}')


# ## 8. Run Experiment 1 — LLM-Only Baseline
# 
# No KB access. The LLM predicts purely from product title and description. For numeric attributes (speeds, dimensions) we expect near-zero accuracy since these values are product-specific and do not appear in descriptions.

# In[9]:


def parse_response(response_text, attribute):
    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith('VALUE:'):
            val = line.split(':',1)[1].strip().strip('"').strip("'")
            return 'UNKNOWN' if val.upper() in {'UNKNOWN','NONE','NAN','NULL',''} else val
    pat = rf'{attribute}\s*[:\u2192>\-]+\s*([^\s|]+)'
    m = re.search(pat, response_text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip('"').strip("'").strip('[]')
        if val.upper() != 'UNKNOWN' and len(val)>2 and val.lower() not in {'none','nan','null','exact','value'}:
            return val
    if attribute in NUMERIC_ATTRIBUTES:
        nums = re.findall(r'\b\d+\.?\d*\b', response_text)
        if nums: return nums[0]
    cleaned = response_text.strip().strip('"').strip("'")
    if cleaned.upper().startswith('VALUE:'): cleaned = cleaned.split(':',1)[1].strip()
    if cleaned and len(cleaned)<80 and '\n' not in cleaned and cleaned.upper() not in {'UNKNOWN','NONE','NULL','NAN',''}:
        return cleaned
    return 'UNKNOWN'

def predict_llm_only(text, attribute, timeout=300):
    result = ['UNKNOWN']
    def target():
        try:
            prompt = FEW_SHOT_PROMPTS_LLM[attribute].format(text=str(text)[:600])
            result[0] = parse_response(predict_model.invoke([HumanMessage(content=prompt)]).content.strip(), attribute)
        except: result[0] = 'UNKNOWN'
    t = threading.Thread(target=target); t.start(); t.join(timeout=timeout)
    if t.is_alive(): print('  ⚠️ TIMEOUT'); return 'UNKNOWN'
    return result[0]


if os.path.exists(LLM_ONLY_RESULT):
    print(f'Loading existing results from {LLM_ONLY_RESULT}...')
    results_df = pd.read_csv(LLM_ONLY_RESULT)
    results_df['predicted'] = results_df['predicted'].apply(fix_prediction)
    results_df['unknown']   = results_df['predicted'] == 'UNKNOWN'
    results_df['correct_standard'] = results_df.apply(
        lambda r: is_correct_standard(r['predicted'], r['ground_truth'], r['attribute']), axis=1)
    results_df['ce_judgment'] = [evaluate_ce(r['predicted'], r['ground_truth'], r['attribute']) for _,r in results_df.iterrows()]
    results_df.to_csv(LLM_ONLY_RESULT, index=False)
else:
    print('Running Exp 1 — LLM-only...')
    t0, predictions = time.time(), []
    for i, (_, task) in enumerate(eval_df.iterrows()):
        idx, attr, gt = task['df1_idx'], task['attribute'], task['ground_truth']
        text      = query_df.loc[idx, 'title_description']
        predicted = predict_llm_only(text, attr)
        eta       = (time.time()-t0)/(i+1) * (len(eval_df)-i-1)
        print(f'  [{i+1}/{len(eval_df)}] Row {idx} | {attr:<22} | GT: {str(gt):<25} | Pred: {predicted:<25} | ETA: {eta/60:.1f}min')
        predictions.append({'df1_idx':idx,'config':'LLM-only','attribute':attr,'is_numeric':task['is_numeric'],
                            'ground_truth':gt,'predicted':predicted,'unknown':predicted=='UNKNOWN'})
    results_df = pd.DataFrame(predictions)
    results_df['correct_standard'] = results_df.apply(lambda r: is_correct_standard(r['predicted'],r['ground_truth'],r['attribute']),axis=1)
    results_df['ce_judgment'] = [evaluate_ce(r['predicted'],r['ground_truth'],r['attribute']) for _,r in results_df.iterrows()]
    results_df.to_csv(LLM_ONLY_RESULT, index=False)
    print(f'Done in {time.time()-t0:.1f}s — saved to {LLM_ONLY_RESULT}')

print_summary(results_df, 'Exp 1: LLM-only')


# ## 9. Text Conversion Functions for Embedding Models
# 
# Each embedding model has a different convention for query vs document encoding. Using the wrong prefix (or no prefix) on a model that requires one can significantly hurt retrieval quality.

# In[10]:


def row_to_text_base(row):
    """Base text: title_description if available, else structured fields + description snippet.
    Used for MiniLM and OpenAI (no prefix needed)."""
    td = row.get('title_description','')
    if pd.notna(td) and str(td).strip(): return str(td).strip()[:400]
    attrs = ['title','model','model_number','brand','product_type']
    text  = ' | '.join([str(row[a]) for a in attrs if pd.notna(row.get(a))])
    desc  = row.get('description','')
    if pd.notna(desc) and str(desc).strip(): text += ' | ' + str(desc).strip()[:200]
    return text

def row_to_text_bge(row, is_query=False):
    """BGE: query prefix 'Represent this product for retrieval: ', no prefix for documents."""
    text = row_to_text_base(row)
    return ('Represent this product for retrieval: ' + text) if is_query else text

def row_to_text_e5(row, is_query=False):
    """E5: 'query: ' for queries, 'passage: ' for documents."""
    text = row_to_text_base(row)
    return ('query: ' + text) if is_query else ('passage: ' + text)

def row_to_text_stella(row, is_query=False):
    """Stella: instruction-based prefix for queries, no prefix for documents."""
    text = row_to_text_base(row)
    return ('Instruct: Retrieve similar products for data cleaning\nQuery: ' + text) if is_query else text

print('Text conversion functions ready.')


# ## 10. Save Embeddings — MiniLM (384-dim, 22M params)

# In[11]:


MINILM_KB_PATH    = f'{EMBEDDINGS_DIR}/minilm_kb.pt'
MINILM_QUERY_PATH = f'{EMBEDDINGS_DIR}/minilm_query.pt'

if os.path.exists(MINILM_KB_PATH):
    print('Loading saved MiniLM embeddings...')
    minilm_kb_embs    = torch.load(MINILM_KB_PATH)
    minilm_query_embs = torch.load(MINILM_QUERY_PATH)
else:
    MINILM_PATH = f'{HF_CACHE}/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
    m = SentenceTransformer(MINILM_PATH)
    t0 = time.time()
    minilm_kb_embs    = m.encode(kb.apply(row_to_text_base,axis=1).tolist(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    minilm_query_embs = m.encode(query_df.apply(row_to_text_base,axis=1).tolist(), convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    torch.save(minilm_kb_embs, MINILM_KB_PATH); torch.save(minilm_query_embs, MINILM_QUERY_PATH)
    print(f'Encoded and saved in {time.time()-t0:.1f}s')

print(f'KB: {minilm_kb_embs.shape} | Query: {minilm_query_embs.shape}')


# ## 11. Save Embeddings — BGE-large-en-v1.5 (1024-dim, 335M params)

# In[12]:


BGE_KB_PATH    = f'{EMBEDDINGS_DIR}/bge_kb.pt'
BGE_QUERY_PATH = f'{EMBEDDINGS_DIR}/bge_query.pt'

if os.path.exists(BGE_KB_PATH):
    print('Loading saved BGE embeddings...')
    bge_kb_embs    = torch.load(BGE_KB_PATH)
    bge_query_embs = torch.load(BGE_QUERY_PATH)
else:
    BGE_SNAPS = glob.glob(f'{HF_CACHE}/models--BAAI--bge-large-en-v1.5/snapshots/*/')
    BGE_PATH  = BGE_SNAPS[0].rstrip('/') if BGE_SNAPS else 'BAAI/bge-large-en-v1.5'
    m = SentenceTransformer(BGE_PATH)
    t0 = time.time()
    bge_kb_embs    = m.encode(kb.apply(lambda r: row_to_text_bge(r,False),axis=1).tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True)
    bge_query_embs = m.encode(query_df.apply(lambda r: row_to_text_bge(r,True),axis=1).tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True)
    torch.save(bge_kb_embs, BGE_KB_PATH); torch.save(bge_query_embs, BGE_QUERY_PATH)
    print(f'Encoded and saved in {time.time()-t0:.1f}s')

print(f'KB: {bge_kb_embs.shape} | Query: {bge_query_embs.shape}')


# ## 12. Save Embeddings — E5-large-v2 (1024-dim, 335M params)

# In[ ]:


E5_KB_PATH    = f'{EMBEDDINGS_DIR}/e5_kb.pt'
E5_QUERY_PATH = f'{EMBEDDINGS_DIR}/e5_query.pt'

if os.path.exists(E5_KB_PATH):
    print('Loading saved E5 embeddings...')
    e5_kb_embs    = torch.load(E5_KB_PATH)
    e5_query_embs = torch.load(E5_QUERY_PATH)
else:
    E5_SNAPS = glob.glob(f'{HF_CACHE}/models--intfloat--e5-large-v2/snapshots/*/')
    E5_PATH  = E5_SNAPS[0].rstrip('/') if E5_SNAPS else 'intfloat/e5-large-v2'
    m = SentenceTransformer(E5_PATH)
    t0 = time.time()
    e5_kb_embs    = m.encode(kb.apply(lambda r: row_to_text_e5(r,False),axis=1).tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True)
    e5_query_embs = m.encode(query_df.apply(lambda r: row_to_text_e5(r,True),axis=1).tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True)
    torch.save(e5_kb_embs, E5_KB_PATH); torch.save(e5_query_embs, E5_QUERY_PATH)
    print(f'Encoded and saved in {time.time()-t0:.1f}s')

print(f'KB: {e5_kb_embs.shape} | Query: {e5_query_embs.shape}')


# ## 13. Save Embeddings — OpenAI text-embedding-3-large (3072-dim, API)
# 
# Requires `OPENAI_API_KEY`. Cost: ~$0.13 per million tokens. Encoding 2,200 KB rows costs less than $0.01.

# In[ ]:


OAI_KB_PATH    = f'{EMBEDDINGS_DIR}/openai_kb.pt'
OAI_QUERY_PATH = f'{EMBEDDINGS_DIR}/openai_query.pt'

if os.path.exists(OAI_KB_PATH):
    print('Loading saved OpenAI embeddings...')
    oai_kb_embs    = torch.load(OAI_KB_PATH)
    oai_query_embs = torch.load(OAI_QUERY_PATH)
    print(f'KB: {oai_kb_embs.shape} | Query: {oai_query_embs.shape}')
else:
    import openai
    client = openai.OpenAI(api_key="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")
    def embed_oai(texts, model='text-embedding-3-large', bs=100):
        embs = []
        for i in range(0, len(texts), bs):
            resp = client.embeddings.create(input=texts[i:i+bs], model=model)
            embs.extend([x.embedding for x in resp.data])
            print(f'  {min(i+bs,len(texts))}/{len(texts)}')
        return torch.tensor(embs, dtype=torch.float32)

    t0 = time.time()
    oai_kb_embs    = embed_oai(kb.apply(row_to_text_base,axis=1).tolist())
    oai_query_embs = embed_oai(query_df.apply(row_to_text_base,axis=1).tolist())
    torch.save(oai_kb_embs, OAI_KB_PATH); torch.save(oai_query_embs, OAI_QUERY_PATH)
    print(f'Done in {time.time()-t0:.1f}s | KB: {oai_kb_embs.shape} | Query: {oai_query_embs.shape}')


# ## 14. Retrieval Quality Comparison — Hit@K Across All Embedding Models
# 
# Hit@K measures recall at the retrieval stage — what fraction of tasks have the correct product cluster in the top-K results. This is the upper bound on accuracy for any downstream LLM extraction step.

# In[ ]:


query_idx_to_pos = {idx: pos for pos, idx in enumerate(query_df.index)}

def hit_at_k(kb_embs, query_embs, k):
    hits = 0
    for _, task in eval_df.iterrows():
        idx        = task['df1_idx']
        cluster_id = query_df.loc[idx,'cluster_id']
        q_emb      = query_embs[query_idx_to_pos[idx]]
        scores     = util.cos_sim(q_emb, kb_embs)[0]
        top_idx    = np.argsort(-scores.cpu().numpy())[:k]
        hits      += int(cluster_id in kb.iloc[top_idx]['cluster_id'].values)
    return hits / len(eval_df)

print(f'Retrieval quality comparison:\n')
print(f'{"Model":<35} {"Hit@3":>8} {"Hit@5":>8} {"Hit@10":>8} {"Hit@20":>8}')
print('-' * 65)

to_compare = [
    ('MiniLM (384-dim)',     minilm_kb_embs, minilm_query_embs),
    ('BGE-large (1024-dim)', bge_kb_embs,    bge_query_embs),
    ('E5-large (1024-dim)',  e5_kb_embs,     e5_query_embs),
]
for path, name, kb_e, q_e in [(OAI_KB_PATH,'OpenAI-3-large',None,None)]:
    if os.path.exists(path):
        kb_e = torch.load(path); q_e = torch.load(path.replace('kb','query'))
        to_compare.append((name, kb_e, q_e))

for name, kb_e, q_e in to_compare:
    h3,h5,h10,h20 = hit_at_k(kb_e,q_e,3), hit_at_k(kb_e,q_e,5), hit_at_k(kb_e,q_e,10), hit_at_k(kb_e,q_e,20)
    print(f'{name:<35} {h3:>8.3f} {h5:>8.3f} {h10:>8.3f} {h20:>8.3f}')

print(f'\nFiles saved:')
for f in sorted(os.listdir(EMBEDDINGS_DIR)):
    print(f'  {f:<40} {os.path.getsize(f"{EMBEDDINGS_DIR}/{f}")/1e6:.1f} MB')

