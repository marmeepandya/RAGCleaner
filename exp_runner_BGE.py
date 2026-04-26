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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

TARGET_ATTRIBUTES  = ['bus_type','model_number','model','read_speed_mb_s','write_speed_mb_s','height_mm','width_mm']
NUMERIC_ATTRIBUTES = {'read_speed_mb_s','write_speed_mb_s','height_mm','width_mm'}
TEXT_ATTRIBUTES    = {'bus_type','model_number','model'}

HF_CACHE       = '/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub'
EMBEDDINGS_DIR = 'embeddings'
RESULTS_DIR    = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Retrieval parameters
TOP_N = 20  # Stage 1: cosine similarity — retrieve this many
TOP_K = 5   # Stage 2: CrossEncoder reranker — keep this many

SKIP_FIELDS = {'id','url','description','title_description','price','priceCurrency','cluster_id'}

print(f'TOP_N={TOP_N} (initial retrieval) | TOP_K={TOP_K} (after reranking)')
print(f'Results will be saved to: {RESULTS_DIR}/')

DATA_DIR = 'normalized_products'
df1      = pd.read_json(f'{DATA_DIR}/dataset_1_normalized.json')
df2      = pd.read_json(f'{DATA_DIR}/dataset_2_normalized.json')
df3      = pd.read_json(f'{DATA_DIR}/dataset_3_normalized.json')
df4      = pd.read_json(f'{DATA_DIR}/dataset_4_normalized.json')
kb_full  = pd.concat([df2, df3, df4], ignore_index=True)
kb       = kb_full.copy()

# Load canonical eval set — must be built by exp_setup.ipynb first
assert os.path.exists('eval_set.csv'), 'Run exp_setup.ipynb first to generate eval_set.csv'
eval_df       = pd.read_csv('eval_set.csv')
query_indices = pd.read_csv('query_indices.csv').iloc[:,0].tolist()
query_df      = df1.loc[query_indices].copy()
query_idx_to_pos = {idx: pos for pos, idx in enumerate(query_df.index)}

print(f'KB: {len(kb):,} rows | Query rows: {len(query_df)} | Eval tasks: {len(eval_df)}')
print('Tasks per attribute:')
print(eval_df['attribute'].value_counts().to_string())

EXP4_FILE       = f'{RESULTS_DIR}/exp4_rag_bge_reranker.csv'
CHECKPOINT_FILE = f'{RESULTS_DIR}/exp4_rag_bge_reranker_checkpoint.csv'
CHECKPOINT_EVERY = 5
TOP_K_BASIC = 3

def load_embeddings(name, kb_path, query_path):
    assert os.path.exists(kb_path),    f'Missing {kb_path} — run exp_setup.ipynb first'
    assert os.path.exists(query_path), f'Missing {query_path} — run exp_setup.ipynb first'
    kb_embs    = torch.load(kb_path)
    query_embs = torch.load(query_path)
    print(f'  {name:<30} KB: {kb_embs.shape} | Query: {query_embs.shape}')
    return kb_embs, query_embs

print('Loading pre-computed embeddings...')
minilm_kb_embs,  minilm_query_embs  = load_embeddings('MiniLM (384-dim)',    f'{EMBEDDINGS_DIR}/minilm_kb.pt',  f'{EMBEDDINGS_DIR}/minilm_query.pt')
bge_kb_embs,     bge_query_embs     = load_embeddings('BGE-large (1024-dim)',f'{EMBEDDINGS_DIR}/bge_kb.pt',     f'{EMBEDDINGS_DIR}/bge_query.pt')
oai_kb_embs,     oai_query_embs     = load_embeddings('OpenAI (3072-dim)',   f'{EMBEDDINGS_DIR}/openai_kb.pt',  f'{EMBEDDINGS_DIR}/openai_query.pt')
print('\nAll embeddings loaded.')

CE_SNAP = glob.glob(f'{HF_CACHE}/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/*/')
CE_PATH = CE_SNAP[0].rstrip('/') if CE_SNAP else 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder = CrossEncoder(CE_PATH)
print(f'CrossEncoder loaded from: {CE_PATH}')

# ── Standard evaluation ───────────────────────────────────────────────────────
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

# ── CE evaluation ─────────────────────────────────────────────────────────────
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

# ── Response parser ───────────────────────────────────────────────────────────
def parse_response(text, attribute):
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith('VALUE:'):
            val = line.split(':',1)[1].strip().strip('"').strip("'")
            return 'UNKNOWN' if val.upper() in {'UNKNOWN','NONE','NAN','NULL',''} else val
    pat = rf'{attribute}\s*[:\u2192>\-]+\s*([^\s|]+)'
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip('"').strip("'").strip('[]')
        if val.upper() != 'UNKNOWN' and len(val)>2 and val.lower() not in {'none','nan','null','exact','value'}:
            return val
    if attribute in NUMERIC_ATTRIBUTES:
        nums = re.findall(r'\b\d+\.?\d*\b', text)
        if nums: return nums[0]
    cleaned = text.strip().strip('"').strip("'")
    if cleaned.upper().startswith('VALUE:'): cleaned = cleaned.split(':',1)[1].strip()
    if cleaned and len(cleaned)<80 and '\n' not in cleaned and cleaned.upper() not in {'UNKNOWN','NONE','NULL','NAN',''}:
        return cleaned
    return 'UNKNOWN'

def fix_prediction(pred):
    if isinstance(pred, str) and pred.strip().upper().startswith('VALUE:'):
        val = pred.strip().split(':',1)[1].strip()
        return 'UNKNOWN' if val.upper() in {'UNKNOWN','NONE','NULL','NAN',''} else val
    return pred

print('Evaluation functions ready.')

def retrieval_metrics(kb_embs, query_embs, k, label=''):
    """Compute Recall@K, Precision@K and NDCG@K across all eval tasks."""
    recall_hits, precision_sum, ndcg_sum = 0, 0.0, 0.0
    per_task = []

    for _, task in eval_df.iterrows():
        idx        = task['df1_idx']
        cluster_id = query_df.loc[idx, 'cluster_id']
        q_emb      = query_embs[query_idx_to_pos[idx]]
        scores     = util.cos_sim(q_emb, kb_embs)[0]
        top_idx    = np.argsort(-scores.cpu().numpy())[:k]
        retrieved  = kb.iloc[top_idx]

        # Which retrieved rows match the correct cluster
        matches    = (retrieved['cluster_id'] == cluster_id).values

        # Recall@K — did we find the correct cluster at all?
        recall_hit = int(matches.any())
        recall_hits += recall_hit

        # Precision@K — what fraction of top-K are correct?
        precision_sum += matches.sum() / k

        # NDCG@K — discount by rank position
        dcg = sum(1.0/math.log2(rank+2) for rank, match in enumerate(matches) if match)
        idcg = sum(1.0/math.log2(rank+2) for rank in range(min(matches.sum(), k)))
        ndcg_sum += (dcg/idcg) if idcg > 0 else 0.0

        per_task.append({'recall': recall_hit, 'precision': matches.sum()/k,
                         'hit': recall_hit, 'cluster_id': cluster_id, 'df1_idx': idx})

    n = len(eval_df)
    results = {
        'recall':    recall_hits / n,
        'precision': precision_sum / n,
        'ndcg':      ndcg_sum / n,
        'per_task':  per_task
    }
    if label:
        print(f'  {label:<35} Recall@{k}: {results["recall"]:.3f}  Precision@{k}: {results["precision"]:.3f}  NDCG@{k}: {results["ndcg"]:.3f}')
    return results

print('Retrieval metric functions ready.')

FEW_SHOT_PROMPTS_RAG = {

'bus_type': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Do NOT use your own knowledge.
If no reference product clearly matches, respond with VALUE:UNKNOWN.
Step 1: Find the reference product that best matches the query product.
Step 2: Copy the bus_type value from that reference product exactly.

Example 1:
Query: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache - WD60EZAZ
Reference products:
  - title: WD Blue 6TB Hard Drive WD60EZAZ | brand: Western Digital | bus_type: SATA III | model_number: WD60EZAZ
Best match: WD Blue 6TB WD60EZAZ → bus_type: SATA III
VALUE:SATA III

Example 2:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference products:
  - title: Corsair Force MP510 960GB NVMe SSD | brand: Corsair | bus_type: PCIe 3.0 x4 | model_number: CSSD-F960GBMP510
Best match: Force MP510 960GB → bus_type: PCIe 3.0 x4
VALUE:PCIe 3.0 x4

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → bus_type: [value from reference]
VALUE:""",

'model_number': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Do NOT generate or guess a model number.
Copy the EXACT model_number from the best matching reference product character by character.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

CRITICAL: model_numbers look like GV-N3080GAMING OC-10GD or CSSD-F960GBMP510.
Pay attention to every character — GV-N166SOC-6GD and GV-N1660OC-6GD are DIFFERENT products.
If you are uncertain between two similar SKUs, respond with VALUE:UNKNOWN.

Example 1:
Query: WD Blue 6TB Desktop Hard Disk Drive - SATA 6Gb/s 256MB Cache 3.5 Inch
Reference products:
  - title: WD Blue 6TB Hard Drive WD60EZAZ | brand: Western Digital | model: WD Blue | model_number: WD60EZAZ
Best match: WD Blue 6TB Hard Drive → model_number: WD60EZAZ
VALUE:WD60EZAZ

Example 2:
Query: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4 NVMe
Reference products:
  - title: Corsair Force MP510 960GB NVMe | brand: Corsair | model: Force Series MP510 | model_number: CSSD-F960GBMP510
Best match: Force MP510 960GB → model_number: CSSD-F960GBMP510
VALUE:CSSD-F960GBMP510

Example 3 (near-identical SKUs):
Query: Gigabyte GeForce GTX 1660 SUPER OC 6G graphics card
Reference products:
  - title: Gigabyte GTX 1660 Ti OC 6G | model_number: GV-N166TOC-6GD | brand: Gigabyte
  - title: Gigabyte GTX 1660 SUPER OC 6G | model_number: GV-N166SOC-6GD | brand: Gigabyte
Best match: GTX 1660 SUPER OC (not Ti) → model_number: GV-N166SOC-6GD
VALUE:GV-N166SOC-6GD

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → model_number: [exact value from reference]
VALUE:""",

'model': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Do NOT use your own knowledge.
Copy the exact model name from the best matching reference product.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache
Reference products:
  - title: WD Blue 6TB Hard Drive WD60EZAZ | brand: Western Digital | model: WD Blue | model_number: WD60EZAZ
Best match: WD Blue 6TB → model: WD Blue
VALUE:WD Blue

Example 2:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe
Reference products:
  - title: Corsair Force MP510 960GB NVMe SSD | brand: Corsair | model: Force Series MP510 | model_number: CSSD-F960GBMP510
Best match: Force MP510 → model: Force Series MP510
VALUE:Force Series MP510

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → model: [value from reference]
VALUE:""",

'read_speed_mb_s': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Copy the exact read_speed_mb_s number.
Return a number only. Do NOT return the write speed.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference products:
  - title: Corsair Force MP510 960GB NVMe | model_number: CSSD-F960GBMP510 | read_speed_mb_s: 3480 | write_speed_mb_s: 3000
Best match: Force MP510 960GB → read_speed_mb_s: 3480
VALUE:3480

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → read_speed_mb_s: [value from reference]
VALUE:""",

'write_speed_mb_s': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Copy the exact write_speed_mb_s number.
Return a number only. Do NOT return the read speed.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference products:
  - title: Corsair Force MP510 960GB NVMe | model_number: CSSD-F960GBMP510 | read_speed_mb_s: 3480 | write_speed_mb_s: 3000
Best match: Force MP510 960GB → write_speed_mb_s: 3000
VALUE:3000

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → write_speed_mb_s: [value from reference]
VALUE:""",

'height_mm': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Copy the exact height_mm number.
Return a number only. Do NOT confuse height with width or length.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: MSI GeForce GTX 1660 Ti GAMING X 6G graphics card
Reference products:
  - title: MSI GTX 1660 Ti GAMING X 6G | model_number: V375-040R | height_mm: 46 | width_mm: 127 | length_mm: 247
Best match: GTX 1660 Ti GAMING X → height_mm: 46
VALUE:46

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → height_mm: [value from reference]
VALUE:""",

'width_mm': """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below. Copy the exact width_mm number.
Return a number only. Do NOT confuse width with height or length.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: MSI GeForce GTX 1660 Ti GAMING X 6G graphics card
Reference products:
  - title: MSI GTX 1660 Ti GAMING X 6G | model_number: V375-040R | height_mm: 46 | width_mm: 127 | length_mm: 247
Best match: GTX 1660 Ti GAMING X → width_mm: 127
VALUE:127

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → width_mm: [value from reference]
VALUE:"""
}

print(f'RAG prompts ready for: {list(FEW_SHOT_PROMPTS_RAG.keys())}')

def format_candidates(candidates):
    """Format KB rows for the LLM prompt — all meaningful fields."""
    lines = []
    for _, c in candidates.iterrows():
        fields = {k: str(v) for k,v in c.items()
                  if k not in SKIP_FIELDS and pd.notna(v)
                  and str(v).strip().lower() not in {'','nan','none'}}
        if fields:
            lines.append('  - ' + ' | '.join(f'{k}: {v}' for k,v in fields.items()))
    return '\n'.join(lines) if lines else '  (no candidates retrieved)'


def retrieve_top_n(q_emb, kb_embs, n):
    """Stage 1: cosine similarity retrieval — returns top-N KB row indices."""
    scores  = util.cos_sim(q_emb, kb_embs)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:n]
    return top_idx


def rerank(query_text, top_n_idx, kb_texts, k):
    """Stage 2: CrossEncoder reranking — returns top-K indices from top-N candidates."""
    cands_texts = [kb_texts[i] for i in top_n_idx]
    pairs       = [[query_text[:300], t[:300]] for t in cands_texts]
    scores      = cross_encoder.predict(pairs)
    top_k_local = np.argsort(-scores)[:k]
    return top_n_idx[top_k_local], scores[top_k_local]


def predict_with_timeout(predict_fn, timeout=300):
    """Run predict_fn in a thread with timeout."""
    result = ['UNKNOWN']
    def target():
        try: result[0] = predict_fn()
        except: result[0] = 'UNKNOWN'
    t = threading.Thread(target=target)
    t.start(); t.join(timeout=timeout)
    if t.is_alive(): print('  ⚠️ TIMEOUT'); return 'UNKNOWN'
    return result[0]


def evaluate_and_save(results_df, config_name, filename):
    """Add evaluation columns, save, and print summary."""
    results_df['predicted']        = results_df['predicted'].apply(fix_prediction)
    results_df['unknown']          = results_df['predicted'] == 'UNKNOWN'
    results_df['correct_standard'] = results_df.apply(
        lambda r: is_correct_standard(r['predicted'], r['ground_truth'], r['attribute']), axis=1)
    results_df['ce_judgment']      = [
        evaluate_ce(r['predicted'], r['ground_truth'], r['attribute'])
        for _, r in results_df.iterrows()]
    path = f'{RESULTS_DIR}/{filename}'
    results_df.to_csv(path, index=False)

    std = results_df['correct_standard'].mean()
    ce  = results_df['ce_judgment'].isin(['correct','acceptable']).mean()
    unk = results_df['unknown'].mean()
    print(f'\n{"="*60}\nRESULTS — {config_name}\n{"="*60}')
    print(f'Standard accuracy:    {std:.3f} ({std*100:.1f}%)')
    print(f'CE eval (c+a):        {ce:.3f} ({ce*100:.1f}%)')
    print(f'UNKNOWN rate:         {unk:.3f} ({unk*100:.1f}%)')
    print(f'Total tasks:          {len(results_df)}')
    print('\nPer-attribute:')
    print(results_df.groupby('attribute').agg(
        n=('correct_standard','count'),
        std_acc=('correct_standard','mean'),
        ce_acc=('ce_judgment', lambda x: x.isin(['correct','acceptable']).mean()),
        unknown=('unknown','mean')
    ).round(3).to_string())
    print(f'\n✓ Saved to {path}')
    return results_df

print('Helper functions ready.')

predict_model = ChatOllama(
    model='llama3.1:8b',
    temperature=0,
    seed=42,
    base_url='http://127.0.0.1:11435'
)
print(f'Ollama: {repr(predict_model.invoke("Say OK").content[:20])}')

EXP4_FILE = 'exp4_rag_bge_reranker.csv'

# BGE KB texts were encoded without prefix — use same base text for CrossEncoder pairs
def row_to_text_bge_doc(row):
    td = row.get('title_description','')
    if pd.notna(td) and str(td).strip(): return str(td).strip()[:400]
    attrs = ['title','model','model_number','brand','product_type']
    text  = ' | '.join([str(row[a]) for a in attrs if pd.notna(row.get(a))])
    desc  = row.get('description','')
    if pd.notna(desc) and str(desc).strip(): text += ' | ' + str(desc).strip()[:200]
    return text

kb_texts_bge = kb.apply(row_to_text_bge_doc, axis=1).tolist()

if os.path.exists(f'{RESULTS_DIR}/{EXP4_FILE}'):
    print(f'Loading existing Exp 4 results...')
    exp4_df = pd.read_csv(f'{RESULTS_DIR}/{EXP4_FILE}')
else:
    print('Running Exp 4 — RAG-BGE-Reranker...')

    print(f'\nRetrieval quality (BGE → CrossEncoder):')
    retrieval_metrics(bge_kb_embs, bge_query_embs, TOP_N, f'BGE@{TOP_N} (before rerank)')

    t0, predictions = time.time(), []
    for i, (_, task) in enumerate(eval_df.iterrows()):
        idx, attr, gt = task['df1_idx'], task['attribute'], task['ground_truth']
        text  = query_df.loc[idx, 'title_description']
        q_emb = bge_query_embs[query_idx_to_pos[idx]]
        # CrossEncoder pairs use plain text (no BGE prefix)
        query_text = row_to_text_bge_doc(query_df.loc[idx])

        top_n_idx     = retrieve_top_n(q_emb, bge_kb_embs, TOP_N)
        top_k_idx, _  = rerank(query_text, top_n_idx, kb_texts_bge, TOP_K)
        candidates    = kb.iloc[top_k_idx]

        def predict():
            prompt = FEW_SHOT_PROMPTS_RAG[attr].format(
                text=str(text)[:500], candidates=format_candidates(candidates))
            return parse_response(predict_model.invoke([HumanMessage(content=prompt)]).content.strip(), attr)

        predicted = predict_with_timeout(predict)
        eta = (time.time()-t0)/(i+1) * (len(eval_df)-i-1)
        print(f'  [{i+1}/{len(eval_df)}] Row {idx} | {attr:<22} | GT: {str(gt):<25} | Pred: {predicted:<25} | ETA: {eta/60:.1f}min')
        predictions.append({'df1_idx':idx,'config':'RAG-BGE-Reranker','attribute':attr,
                            'is_numeric':task['is_numeric'],'ground_truth':gt,'predicted':predicted,'unknown':predicted=='UNKNOWN'})

        if len(predictions) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(predictions).to_csv(CHECKPOINT_FILE, index=False)
            print(f'  ✓ Checkpoint saved ({len(predictions)}/{len(eval_df)} done)')
        
   
     # Final save and evaluate
    pd.DataFrame(predictions).to_csv(CHECKPOINT_FILE, index=False)
    exp4_df = evaluate_and_save(pd.DataFrame(predictions), 'Exp 4: RAG-BGE-Reranker', 'exp4_rag_bge_reranker.csv')
    os.remove(CHECKPOINT_FILE)  # clean up checkpoint once final file exists
    print('Checkpoint removed — final file saved.')