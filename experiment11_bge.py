"""
Experiment 11 — BGE-large-en-v1.5 Embeddings + CrossEncoder Re-ranking
=======================================================================
Key difference from Experiment 10:
- Replaces all-MiniLM-L6-v2 with BAAI/bge-large-en-v1.5 for KB and query embeddings
- BGE requires a query prefix for retrieval queries
- Everything else identical to Experiment 10 (clean KB, CrossEncoder reranker, same eval)

Pipeline:
  Query (BGE prefix) → BGE embed → top-20 → CrossEncoder rerank → top-3 → Llama predict
                                                                          ↓
                                            CrossEncoder (text eval) ← prediction
                                            Rule-based 10% tol (numeric eval) ←
"""

import sys
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib64/python3.12/site-packages')
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib/python3.12/site-packages')
sys.path.append('/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI')

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import random
import time
import threading
import glob

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer, CrossEncoder, util

random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_ATTRIBUTES = [
    "bus_type", "model_number", "model",
    "read_speed_mb_s", "write_speed_mb_s",
    "height_mm", "width_mm"
]
NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}
TEXT_ATTRIBUTES    = {"bus_type", "model_number", "model"}

TOP_N = 20   # MiniLM retrieves this many
TOP_K = 3    # CrossEncoder re-ranks to this many

RESULT_FILE        = "results_exp11_bge_rag.csv"
FINAL_RESULTS_FILE = "results_exp11_bge_all.csv"

print("=" * 60)
print("Experiment 11 — BGE-large + CrossEncoder Re-ranking")
print("=" * 60)

# ── 1. Load datasets ──────────────────────────────────────────────────────────
print("\n[1/8] Loading datasets...")
df1     = pd.read_json("normalized_products/dataset_1_normalized.json")
df2     = pd.read_json("normalized_products/dataset_2_normalized.json")
df3     = pd.read_json("normalized_products/dataset_3_normalized.json")
df4     = pd.read_json("normalized_products/dataset_4_normalized.json")
kb_full = pd.concat([df2, df3, df4], ignore_index=True)
print(f"  Dataset 1: {len(df1)} rows")
print(f"  KB full:   {len(kb_full)} rows")

# ── 2. Build eval set (same 25 rows as all other experiments) ─────────────────
print("\n[2/8] Building evaluation set...")

def get_ground_truth(cluster_id, attribute):
    matches = kb_full[kb_full["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

MIN_PER_ATTR = 5
all_candidates = []
for idx, row in df1.iterrows():
    missing_attrs = []
    for attr in TARGET_ATTRIBUTES:
        if pd.isna(row.get(attr)):
            gt = get_ground_truth(row["cluster_id"], attr)
            if gt is not None:
                missing_attrs.append((attr, gt))
    if missing_attrs:
        all_candidates.append({"df1_idx": idx, "missing_attrs": missing_attrs})

attr_counts  = {attr: 0 for attr in TARGET_ATTRIBUTES}
selected_rows = []
selected_idx  = set()

for candidate in all_candidates:
    if len(selected_rows) >= 25:
        break
    contributes = any(attr_counts[attr] < MIN_PER_ATTR for attr, _ in candidate["missing_attrs"])
    if contributes:
        selected_rows.append(candidate)
        selected_idx.add(candidate["df1_idx"])
        for attr, _ in candidate["missing_attrs"]:
            attr_counts[attr] += 1

for candidate in all_candidates:
    if len(selected_rows) >= 25:
        break
    if candidate["df1_idx"] not in selected_idx:
        selected_rows.append(candidate)
        selected_idx.add(candidate["df1_idx"])
        for attr, _ in candidate["missing_attrs"]:
            attr_counts[attr] += 1

eval_records = []
for item in selected_rows:
    for attr, gt in item["missing_attrs"]:
        eval_records.append({
            "df1_idx":    item["df1_idx"],
            "attribute":  attr,
            "ground_truth": gt,
            "is_numeric": attr in NUMERIC_ATTRIBUTES
        })

eval_df      = pd.DataFrame(eval_records)
query_indices = [item["df1_idx"] for item in selected_rows]
query_df     = df1.loc[query_indices].copy()
kb           = kb_full.copy()

print(f"  Query rows:       {len(query_df)}")
print(f"  Total eval tasks: {len(eval_df)}")
print(f"  KB size:          {len(kb)} rows")

# ── 3. LLM setup ─────────────────────────────────────────────────────────────
print("\n[3/8] Connecting to Ollama...")
predict_model = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11435")
test = predict_model.invoke("Say OK")
print(f"  Ollama OK: {repr(test.content[:20])}")

# ── 4. Retriever setup — BGE-large + CrossEncoder ─────────────────────────────
print("\n[4/8] Loading BGE-large-en-v1.5 embedding model...")

BGE_CACHE = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/"
snapshots = glob.glob(BGE_CACHE + "*/")
if snapshots:
    BGE_PATH = snapshots[0].rstrip("/")
    print(f"  Found BGE at: {BGE_PATH}")
else:
    BGE_PATH = "BAAI/bge-large-en-v1.5"
    print("  Using BGE by name (must be cached)")

embedding_model = SentenceTransformer(BGE_PATH)
print("  BGE-large loaded.")

# BGE requires query prefix for retrieval
def row_to_text(row, is_query=False):
    attrs = ["title", "model", "model_number", "brand", "product_type"]
    text  = " | ".join([str(row[a]) for a in attrs if pd.notna(row.get(a))])
    if is_query:
        return "Represent this product for retrieval: " + text
    return text

print("  Encoding KB with BGE-large...")
kb_texts      = kb.apply(lambda r: row_to_text(r, is_query=False), axis=1).tolist()
kb_embeddings = embedding_model.encode(
    kb_texts, convert_to_tensor=True,
    batch_size=32, show_progress_bar=True
)
print(f"  KB encoded: {len(kb_texts)} rows")

print("  Encoding query rows with BGE-large (with prefix)...")
query_texts      = query_df.apply(lambda r: row_to_text(r, is_query=True), axis=1).tolist()
query_embeddings = embedding_model.encode(
    query_texts, convert_to_tensor=True,
    batch_size=32, show_progress_bar=True
)
print("  Query rows encoded.")

query_idx_to_pos = {idx: pos for pos, idx in enumerate(query_df.index)}

# CrossEncoder for re-ranking and text evaluation
print("\n  Loading CrossEncoder re-ranker...")
CE_CACHE  = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/"
ce_snaps  = glob.glob(CE_CACHE + "*/")
CE_PATH   = ce_snaps[0].rstrip("/") if ce_snaps else "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(CE_PATH)
print(f"  CrossEncoder loaded from: {CE_PATH}")
print(f"\n  Retrieval pipeline: BGE top-{TOP_N} → CrossEncoder rerank → top-{TOP_K}")

# ── 5. Retrieve and re-rank ───────────────────────────────────────────────────
def retrieve_and_rerank(query_text_plain, q_emb, top_n=TOP_N, top_k=TOP_K):
    """Stage 1: BGE cosine similarity → top_n.
       Stage 2: CrossEncoder rerank → top_k."""
    scores    = util.cos_sim(q_emb, kb_embeddings)[0]
    top_n_idx = np.argsort(-scores.cpu().numpy())[:top_n]
    cands_n   = kb.iloc[top_n_idx].copy()
    cands_n_texts = [kb_texts[i] for i in top_n_idx]

    pairs         = [[query_text_plain[:300], t[:300]] for t in cands_n_texts]
    rerank_scores = cross_encoder.predict(pairs)
    top_k_local   = np.argsort(-rerank_scores)[:top_k]

    return cands_n.iloc[top_k_local], rerank_scores[top_k_local]

# ── 6. Prompts ────────────────────────────────────────────────────────────────
print("\n[5/8] Defining prompts...")

FEW_SHOT_PROMPTS_RAG = {

"bus_type": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. If no reference product clearly matches, respond with VALUE:UNKNOWN.
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

"model_number": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. Do NOT generate or guess a model number.
Copy the EXACT model_number from the best matching reference product character by character.
If no reference product clearly matches, respond with VALUE:UNKNOWN.
Step 1: Find the reference product that best matches the query (same brand, same product line, same specs).
Step 2: Copy the model_number from that reference product exactly.

Example 1:
Query: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache 3.5 Inch
Reference products:
  - title: WD Blue 6TB Hard Drive WD60EZAZ | brand: Western Digital | model: WD Blue | model_number: WD60EZAZ | bus_type: SATA III
Best match: WD Blue 6TB Hard Drive → model_number: WD60EZAZ
VALUE:WD60EZAZ

Example 2:
Query: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4 NVMe
Reference products:
  - title: Corsair Force MP510 960GB NVMe | brand: Corsair | model: Force Series MP510 | model_number: CSSD-F960GBMP510 | read_speed_mb_s: 3480
Best match: Force MP510 960GB → model_number: CSSD-F960GBMP510
VALUE:CSSD-F960GBMP510

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → model_number: [exact value from reference]
VALUE:""",

"model": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. Copy the exact model name from the best matching reference product.
If no reference product clearly matches, respond with VALUE:UNKNOWN.
Step 1: Find the reference product that best matches the query product.
Step 2: Copy the model value from that reference product exactly.

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

"read_speed_mb_s": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. Copy the exact read_speed_mb_s number from the best matching reference.
Return a number only. Do NOT return the write speed.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference products:
  - title: Corsair Force MP510 960GB NVMe | model_number: CSSD-F960GBMP510 | read_speed_mb_s: 3480 | write_speed_mb_s: 3000
Best match: Force MP510 960GB → read_speed_mb_s: 3480
VALUE:3480

Example 2:
Query: Samsung 860 EVO 4TB 2.5 inch SATA SSD
Reference products:
  - title: Samsung 860 EVO 4TB SSD | model_number: MZ-76E4T0B | read_speed_mb_s: 550 | write_speed_mb_s: 520
Best match: Samsung 860 EVO 4TB → read_speed_mb_s: 550
VALUE:550

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → read_speed_mb_s: [value from reference]
VALUE:""",

"write_speed_mb_s": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. Copy the exact write_speed_mb_s number from the best matching reference.
Return a number only. Do NOT return the read speed.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: CORSAIR Force Series MP510 960GB M.2 SSD NVMe PCIe Gen3
Reference products:
  - title: Corsair Force MP510 960GB NVMe | model_number: CSSD-F960GBMP510 | read_speed_mb_s: 3480 | write_speed_mb_s: 3000
Best match: Force MP510 960GB → write_speed_mb_s: 3000
VALUE:3000

Example 2:
Query: Samsung 860 EVO 4TB 2.5 inch SATA SSD
Reference products:
  - title: Samsung 860 EVO 4TB SSD | model_number: MZ-76E4T0B | read_speed_mb_s: 550 | write_speed_mb_s: 520
Best match: Samsung 860 EVO 4TB → write_speed_mb_s: 520
VALUE:520

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → write_speed_mb_s: [value from reference]
VALUE:""",

"height_mm": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. Copy the exact height_mm number from the best matching reference.
Return a number only. Do NOT confuse height with width or length.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: MSI GeForce GTX 1660 Ti GAMING X 6G graphics card
Reference products:
  - title: MSI GTX 1660 Ti GAMING X 6G | model_number: V375-040R | height_mm: 46 | width_mm: 127 | length_mm: 247
Best match: GTX 1660 Ti GAMING X → height_mm: 46
VALUE:46

Example 2:
Query: Samsung T5 1TB External Portable SSD
Reference products:
  - title: Samsung T5 1TB Portable SSD | model_number: MU-PA1T0B | height_mm: 10.5 | width_mm: 57
Best match: Samsung T5 1TB → height_mm: 10.5
VALUE:10.5

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → height_mm: [value from reference]
VALUE:""",

"width_mm": """\
You are a product data expert filling missing values in a product database.
You MUST use ONLY the reference products below to find the answer.
Do NOT use your own knowledge. Copy the exact width_mm number from the best matching reference.
Return a number only. Do NOT confuse width with height or length.
If no reference product clearly matches, respond with VALUE:UNKNOWN.

Example 1:
Query: MSI GeForce GTX 1660 Ti GAMING X 6G graphics card
Reference products:
  - title: MSI GTX 1660 Ti GAMING X 6G | model_number: V375-040R | height_mm: 46 | width_mm: 127 | length_mm: 247
Best match: GTX 1660 Ti GAMING X → width_mm: 127
VALUE:127

Example 2:
Query: Samsung T5 1TB External Portable SSD
Reference products:
  - title: Samsung T5 1TB Portable SSD | model_number: MU-PA1T0B | height_mm: 10.5 | width_mm: 57
Best match: Samsung T5 1TB → width_mm: 57
VALUE:57

Now fill the missing value:
Query: {text}
Reference products:
{candidates}
Best match: [identify matching product] → width_mm: [value from reference]
VALUE:"""
}

print(f"  Prompts defined for: {list(FEW_SHOT_PROMPTS_RAG.keys())}")

# ── 7. Format candidates ──────────────────────────────────────────────────────
SKIP_FIELDS = {"id", "url", "description", "title_description", "price", "priceCurrency", "cluster_id"}

def format_candidates(candidates):
    lines = []
    for _, c in candidates.iterrows():
        fields = {
            k: str(v) for k, v in c.items()
            if k not in SKIP_FIELDS
            and pd.notna(v)
            and str(v).strip().lower() not in {"", "nan", "none"}
        }
        if fields:
            lines.append("  - " + " | ".join(f"{k}: {v}" for k, v in fields.items()))
    return "\n".join(lines) if lines else "  (no candidates retrieved)"

# ── 8. Retrieval Hit@k analysis ───────────────────────────────────────────────
print("\n[6/8] Retrieval quality analysis (BGE Hit@20 vs CrossEncoder Hit@3)...")
print(f"=== RETRIEVAL QUALITY: BGE Hit@{TOP_N} vs CrossEncoder Hit@{TOP_K} ===\n")

hit_bge      = 0
hit_crossenc = 0
total_tasks  = 0

for _, task in eval_df.iterrows():
    idx        = task["df1_idx"]
    cluster_id = query_df.loc[idx, "cluster_id"]
    pos        = query_idx_to_pos[idx]
    q_emb      = query_embeddings[pos]
    # use plain text (no prefix) for CrossEncoder pairs
    query_text_plain = row_to_text(query_df.loc[idx], is_query=False)

    scores    = util.cos_sim(q_emb, kb_embeddings)[0]
    top_n_idx = np.argsort(-scores.cpu().numpy())[:TOP_N]
    cands_n   = kb.iloc[top_n_idx]
    bge_hit   = cluster_id in cands_n["cluster_id"].values

    cands_k, _ = retrieve_and_rerank(query_text_plain, q_emb)
    ce_hit     = cluster_id in cands_k["cluster_id"].values

    hit_bge      += int(bge_hit)
    hit_crossenc += int(ce_hit)
    total_tasks  += 1

    b = "✓" if bge_hit else "✗"
    c = "✓" if ce_hit  else "✗"
    print(f"  Row {idx:<4} | cluster {cluster_id} | BGE@{TOP_N}: {b} | CrossEnc@{TOP_K}: {c}")

print(f"\nBGE Hit@{TOP_N}:          {hit_bge}/{total_tasks} = {hit_bge/total_tasks:.3f}")
print(f"CrossEncoder Hit@{TOP_K}:  {hit_crossenc}/{total_tasks} = {hit_crossenc/total_tasks:.3f}")
print(f"Re-ranking change:         {hit_crossenc - hit_bge:+d} tasks")
print()

# ── 9. Prediction ─────────────────────────────────────────────────────────────
def predict_attribute(product_text, attribute, candidates=None):
    if candidates is not None:
        cands_text      = format_candidates(candidates)
        prompt_template = FEW_SHOT_PROMPTS_RAG[attribute]
        prompt          = prompt_template.format(
            text=str(product_text)[:500],
            candidates=cands_text
        )
    else:
        return "UNKNOWN"

    response      = predict_model.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith("VALUE:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if value.upper() != "UNKNOWN" and value.lower() not in {"", "none", "nan", "null"}:
                return value
            return "UNKNOWN"
    cleaned = response_text.strip().strip('"').strip("'")
    if cleaned and len(cleaned) < 80 and "\n" not in cleaned:
        if cleaned.upper() not in {"UNKNOWN", "NONE", "NULL", "NAN", ""}:
            return cleaned
    return "UNKNOWN"


def predict_with_timeout(product_text, attribute, candidates=None, timeout=300):
    result = ["UNKNOWN"]
    def target():
        try:
            result[0] = predict_attribute(product_text, attribute, candidates)
        except:
            result[0] = "UNKNOWN"
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        print(f"    ⚠️  TIMEOUT — skipping")
        return "UNKNOWN"
    return result[0]


print(f"\n[7/8] Running prediction (BGE top-{TOP_N} → CrossEncoder top-{TOP_K} → Llama)...")
t0          = time.time()
total       = len(eval_df)
predictions = []

for i, (_, task) in enumerate(eval_df.iterrows()):
    idx  = task["df1_idx"]
    attr = task["attribute"]
    gt   = task["ground_truth"]
    text = query_df.loc[idx, "title_description"]

    pos              = query_idx_to_pos[idx]
    q_emb            = query_embeddings[pos]
    query_text_plain = row_to_text(query_df.loc[idx], is_query=False)
    candidates, _    = retrieve_and_rerank(query_text_plain, q_emb)

    predicted = predict_with_timeout(text, attr, candidates, timeout=300)

    elapsed = time.time() - t0
    eta     = (elapsed / (i + 1)) * (total - i - 1)
    print(f"  [{i+1}/{total}] Row {idx} | {attr:<20} | "
          f"GT: {str(gt):<25} | Pred: {predicted:<25} | ETA: {eta/60:.1f}min")

    predictions.append({
        "df1_idx":      idx,
        "config":       "RAG+BGE+Reranker",
        "attribute":    attr,
        "is_numeric":   task["is_numeric"],
        "ground_truth": gt,
        "predicted":    predicted,
        "unknown":      predicted == "UNKNOWN"
    })

results_df = pd.DataFrame(predictions)
print(f"\nDone in {time.time()-t0:.1f}s")

# ── 10. Evaluation ─────────────────────────────────────────────────────────────
print("\n[8/8] Evaluating predictions...")

def is_correct_standard(predicted, ground_truth, attribute):
    if not predicted or str(predicted).strip().lower() in {"", "nan", "none", "unknown", "null"}:
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(",", "").strip())
            g = float(str(ground_truth).replace(",", "").strip())
            return abs(p - g) / abs(g) <= 0.10 if g != 0 else p == 0
        except:
            pass
    p = str(predicted).lower().strip()
    g = str(ground_truth).lower().strip()
    return p == g or p in g or g in p

def evaluate_prediction(predicted, ground_truth, attribute):
    if predicted == "UNKNOWN" or str(predicted).lower() in {"nan", "none", "null", ""}:
        return "wrong"
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(",", "").strip())
            g = float(str(ground_truth).replace(",", "").strip())
            if g == 0:
                return "correct" if p == 0 else "wrong"
            ratio = abs(p - g) / abs(g)
            if ratio <= 0.10: return "correct"
            if ratio <= 0.30: return "acceptable"
            return "wrong"
        except:
            return "wrong"
    else:
        score = cross_encoder.predict([[ground_truth, predicted]])[0]
        if score > 2.0:  return "correct"
        if score > -1.0: return "acceptable"
        return "wrong"

results_df["correct_standard"] = results_df.apply(
    lambda row: is_correct_standard(row["predicted"], row["ground_truth"], row["attribute"]),
    axis=1
)

judgments = []
for _, row in results_df.iterrows():
    j = evaluate_prediction(row["predicted"], row["ground_truth"], row["attribute"])
    judgments.append(j)
    print(f"  {row['attribute']:<20} | GT: {str(row['ground_truth']):<25} | "
          f"Pred: {str(row['predicted']):<25} | {j}")

results_df["ce_judgment"] = judgments
results_df.to_csv(RESULT_FILE, index=False)

# ── Final summary ──────────────────────────────────────────────────────────────
standard_acc  = results_df["correct_standard"].mean()
ce_correct    = (results_df["ce_judgment"] == "correct").mean()
ce_acceptable = (results_df["ce_judgment"].isin(["correct", "acceptable"])).mean()

print("\n" + "=" * 60)
print("RESULTS — Exp 11: BGE-large + CrossEncoder Re-ranking (clean KB)")
print("=" * 60)
print(f"{'Standard accuracy':<45} {standard_acc:.3f}")
print(f"{'CE eval — correct only':<45} {ce_correct:.3f}")
print(f"{'CE eval — correct + acceptable':<45} {ce_acceptable:.3f}")
print(f"{'UNKNOWN rate':<45} {results_df['unknown'].mean():.3f}")
print(f"{'Total tasks':<45} {len(results_df)}")

print("\nPer-attribute breakdown:")
per_attr = results_df.groupby("attribute").apply(lambda x: pd.Series({
    "total":                len(x),
    "standard_acc":         x["correct_standard"].mean(),
    "ce_correct+acceptable":(x["ce_judgment"].isin(["correct", "acceptable"])).mean(),
    "unknown_rate":         x["unknown"].mean()
}), include_groups=False).round(3)
print(per_attr.to_string())

print("\nCE judgment distribution:")
print(results_df["ce_judgment"].value_counts())
print(f"\n✓ Saved to {RESULT_FILE}")

# Comparison with Exp 10 if available
if os.path.exists("results_exp10_rag.csv"):
    exp10 = pd.read_csv("results_exp10_rag.csv")
    print("\n" + "=" * 60)
    print("COMPARISON: Exp 10 (MiniLM) vs Exp 11 (BGE-large)")
    print("=" * 60)
    print(f"{'Config':<35} {'Standard':>10} {'CE eval':>10} {'UNKNOWN':>10}")
    print("-" * 65)
    print(f"{'Exp10 MiniLM+CrossEncoder':<35} "
          f"{exp10['correct_standard'].mean():>10.3f} "
          f"{(exp10['ce_judgment'].isin(['correct','acceptable'])).mean():>10.3f} "
          f"{exp10['unknown'].mean():>10.3f}")
    print(f"{'Exp11 BGE+CrossEncoder':<35} "
          f"{standard_acc:>10.3f} "
          f"{ce_acceptable:>10.3f} "
          f"{results_df['unknown'].mean():>10.3f}")

print("\nDone.")
