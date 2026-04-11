#!/usr/bin/env python
# coding: utf-8

# # Experiment 10 — RAG with CrossEncoder Re-ranking + Improved Evaluation
# 
# **Key improvements over experiment 8.1:**
# - Two-stage retrieval: top-20 by MiniLM cosine similarity → re-ranked to top-3 by CrossEncoder
# - CrossEncoder used for evaluation of text attributes (removes same-model bias)
# - Numeric attributes evaluated by rule only (10% tolerance, no LLM needed)
# - Llama 3.1 8B used only for prediction (one job, not three)
# 
# **Pipeline:**
# ```
# Query → MiniLM (embed) → top-20 candidates → CrossEncoder (rerank) → top-3 → Llama (predict)
#                                                                                ↓
#                                               CrossEncoder (eval text) ← prediction
#                                               Rule-based (eval numeric) ←
# ```
# 
# **Setup:**
# - Same 25 query rows, same KB (full 2,200 rows) as experiment 8.1
# - TOP_N=20 for initial retrieval, TOP_K=3 after re-ranking
# - CrossEncoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (must be pre-downloaded)

# ## 1. Setup

# In[18]:


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

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer, CrossEncoder, util

random.seed(42)
np.random.seed(42)

TARGET_ATTRIBUTES = [
    "bus_type", "model_number", "model",
    "read_speed_mb_s", "write_speed_mb_s",
    "height_mm", "width_mm"
]
NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}
TEXT_ATTRIBUTES = {"bus_type", "model_number", "model"}

USE_RAG = True  # Set to False for LLM-only baseline

LLM_FILE  = "results_exp10_easy_llm_only.csv"
RAG_FILE  = "results_exp10_easy_rag.csv"
RESULT_FILE = RAG_FILE if USE_RAG else LLM_FILE
FINAL_RESULTS_FILE = "results_exp10_easy_all.csv"

TOP_N = 20   # candidates retrieved by MiniLM
TOP_K = 3    # candidates passed to LLM after re-ranking

print(f"Setup complete. USE_RAG={USE_RAG}")


# ## 2. Load Datasets

# In[19]:


df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
df2 = pd.read_json("normalized_products/dataset_2_normalized.json")
df3 = pd.read_json("normalized_products/dataset_3_normalized.json")
df4 = pd.read_json("normalized_products/dataset_4_normalized.json")
kb_full = pd.concat([df2, df3, df4], ignore_index=True)

print(f"Dataset 1: {len(df1)} rows")
print(f"KB full:   {len(kb_full)} rows")


# ## 3. Build Evaluation Set (same 25 rows as exp 8.1)

# In[20]:


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

attr_counts = {attr: 0 for attr in TARGET_ATTRIBUTES}
selected_rows = []
selected_idx = set()

for candidate in all_candidates:
    if len(selected_rows) >= 25:
        break
    contributes = any(
        attr_counts[attr] < MIN_PER_ATTR
        for attr, _ in candidate["missing_attrs"]
    )
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
            "df1_idx": item["df1_idx"],
            "attribute": attr,
            "ground_truth": gt,
            "is_numeric": attr in NUMERIC_ATTRIBUTES
        })

eval_df = pd.DataFrame(eval_records)
query_indices = [item["df1_idx"] for item in selected_rows]
query_df = df1.loc[query_indices].copy()

print(f"Query rows:       {len(query_df)}")
print(f"Total eval tasks: {len(eval_df)}")
print()
print("Tasks per attribute:")
print(eval_df["attribute"].value_counts())
print()
print("Attribute coverage check:")
for attr in TARGET_ATTRIBUTES:
    count = (eval_df["attribute"] == attr).sum()
    status = "✓" if count >= MIN_PER_ATTR else "✗ NEED MORE"
    print(f"  {attr:<25} {count:>3} instances  {status}")


# ## 4. Build Knowledge Base (full 2,200 rows)

# In[ ]:


kb = kb_full.copy()
print(f"KB size: {len(kb)} rows")


# ## 5. Sanity Check

# In[21]:


print("=== SANITY CHECK: Ground truth present in KB ===\n")
found = 0
not_found = 0
for _, task in eval_df.iterrows():
    idx = task["df1_idx"]
    attr = task["attribute"]
    gt = task["ground_truth"]
    cluster_id = query_df.loc[idx, "cluster_id"]
    kb_matches = kb[kb["cluster_id"] == cluster_id]
    val_in_kb = any(
        pd.notna(r.get(attr)) and str(r.get(attr)).strip() == str(gt).strip()
        for _, r in kb_matches.iterrows()
    )
    status = "✓" if val_in_kb else "✗"
    found += int(val_in_kb)
    not_found += int(not val_in_kb)
    print(f"{status} Row {idx} | {attr:<20} | GT: {str(gt):<30} | KB matches: {len(kb_matches)}")

print(f"\nCoverage: {found}/{found+not_found} = {100*found/(found+not_found):.1f}%")


# ## 6. Model Setup
# 
# Three separate models, each with a single job:
# - **MiniLM** — initial retrieval (embedding similarity)
# - **CrossEncoder** — re-ranking top-20 to top-3 + evaluating text predictions
# - **Llama 3.1 8B** — prediction only

# In[22]:


# Prediction model — Llama only used for prediction
predict_model = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11435")
test = predict_model.invoke("Say OK")
print("Ollama OK:", repr(test.content[:20]))


# ## 7. Retriever Setup — MiniLM + CrossEncoder

# In[23]:


# ── MiniLM for initial embedding retrieval ───────────────────────────────────
LOCAL_MINILM_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

print("Loading MiniLM embedding model...")
embedding_model = SentenceTransformer(LOCAL_MINILM_PATH)

# ── CrossEncoder for re-ranking and text evaluation ──────────────────────────
# Must be pre-downloaded on login node:
# python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
LOCAL_CROSSENCODER_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/"

# Find the actual snapshot hash
import glob
snapshots = glob.glob(LOCAL_CROSSENCODER_PATH + "*/")
if snapshots:
    LOCAL_CROSSENCODER_PATH = snapshots[0].rstrip("/")
    print(f"Found CrossEncoder at: {LOCAL_CROSSENCODER_PATH}")
else:
    # Fallback: try loading directly by name (will use cache if available)
    LOCAL_CROSSENCODER_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print("Using CrossEncoder by name (must be cached)")

print("Loading CrossEncoder re-ranker...")
cross_encoder = CrossEncoder(LOCAL_CROSSENCODER_PATH)
print("CrossEncoder loaded.")

# ── Encode KB and query rows ─────────────────────────────────────────────────
def row_to_text(row):
    attrs = ["title", "model", "model_number", "brand", "product_type"]
    return " | ".join([str(row[a]) for a in attrs if pd.notna(row.get(a))])

print("Encoding KB with MiniLM...")
kb_texts = kb.apply(row_to_text, axis=1).tolist()
kb_embeddings = embedding_model.encode(
    kb_texts, convert_to_tensor=True,
    batch_size=64, show_progress_bar=True
)
print(f"KB encoded: {len(kb_texts)} rows")

print("Encoding query rows...")
query_texts = query_df.apply(row_to_text, axis=1).tolist()
query_embeddings = embedding_model.encode(
    query_texts, convert_to_tensor=True,
    batch_size=64, show_progress_bar=True
)
print("Query rows encoded.")

query_idx_to_pos = {idx: pos for pos, idx in enumerate(query_df.index)}

print(f"\nRetrieval pipeline: MiniLM top-{TOP_N} → CrossEncoder re-rank → top-{TOP_K}")


# ## 8. Two-Stage Retrieval Function

# In[24]:


def retrieve_and_rerank(query_text, q_emb, top_n=TOP_N, top_k=TOP_K):
    """Stage 1: MiniLM cosine similarity → top_n candidates.
       Stage 2: CrossEncoder re-ranking → top_k final candidates."""
    # Stage 1 — MiniLM
    scores = util.cos_sim(q_emb, kb_embeddings)[0]
    top_n_idx = np.argsort(-scores.cpu().numpy())[:top_n]
    candidates_n = kb.iloc[top_n_idx].copy()
    candidates_n_texts = kb_texts[i] if False else [kb_texts[i] for i in top_n_idx]

    # Stage 2 — CrossEncoder re-ranking
    pairs = [[query_text[:300], cand_text[:300]] for cand_text in candidates_n_texts]
    rerank_scores = cross_encoder.predict(pairs)
    top_k_local_idx = np.argsort(-rerank_scores)[:top_k]

    final_candidates = candidates_n.iloc[top_k_local_idx]
    return final_candidates, rerank_scores[top_k_local_idx]

print("retrieve_and_rerank() defined.")

# Quick test
test_pos = 0
test_text = query_texts[test_pos]
test_emb = query_embeddings[test_pos]
test_cands, test_scores = retrieve_and_rerank(test_text, test_emb)
print(f"\nTest retrieval for: {test_text[:80]}")
print(f"Top-{TOP_K} after re-ranking:")
for i, (_, row) in enumerate(test_cands.iterrows()):
    print(f"  [{i+1}] score={test_scores[i]:.3f} | {row.get('title','')[:70]}")


# ## 9. LLM-Only Prompts

# In[25]:


FEW_SHOT_PROMPTS = {

"bus_type": """\
You are a product data expert. Extract ONLY the bus_type from the product text.
bus_type is the interface/connection type of the product (e.g. PCIe 3.0 x16, SATA III, USB 3.0).
Respond with VALUE:<answer> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache 3.5 Inch - WD60EZAZ
VALUE:SATA III

Example 2:
Product: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4, NVMe, up to 3480/3000MB/s
VALUE:PCIe 3.0 x4

Example 3:
Product: Kingston DataTraveler Vault Privacy 3.0 16GB. Description: USB 3.0 interface, hardware encryption
VALUE:USB 3.0

Now extract:
Product: {text}
VALUE:""",

"model_number": """\
You are a product data expert. Extract ONLY the model_number from the product text.
model_number is the exact manufacturer SKU or part number — usually alphanumeric with dashes,
often found in parentheses or at the end of the product title.
Respond with VALUE:<answer> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache 3.5 Inch - WD60EZAZ
VALUE:WD60EZAZ

Example 2:
Product: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4 (CSSD-F960GBMP510)
VALUE:CSSD-F960GBMP510

Example 3:
Product: Asus Dual GeForce GTX 1650 OC 4GB Graphics Card | 90YV0CV2-M0NA00
VALUE:90YV0CV2-M0NA00

Now extract:
Product: {text}
VALUE:""",

"model": """\
You are a product data expert. Extract ONLY the model name from the product text.
model is the product model name (not the full product title, not the SKU).
Examples: WD Blue, Force Series MP510, Exos 7E8, GeForce RTX 3080 GAMING OC.
Respond with VALUE:<answer> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: WD Blue 6TB Desktop Hard Disk Drive - 5400 RPM SATA 6Gb/s 256MB Cache 3.5 Inch - WD60EZAZ
VALUE:WD Blue

Example 2:
Product: CORSAIR Force Series MP510 960GB M.2 SSD PCIe Gen3 x4, NVMe, up to 3480/3000MB/s
VALUE:Force Series MP510

Example 3:
Product: 4TB Exos 7E8 ST4000NM0115, 7200 RPM, SATA 6Gb/s, 512e, 128MB cache, 3.5-Inch HDD
VALUE:Exos 7E8

Now extract:
Product: {text}
VALUE:""",

"read_speed_mb_s": """\
You are a product data expert. Extract ONLY the sequential READ speed in MB/s from the product text.
Return the number only (e.g. 3480, 550, 3400). Do NOT return write speed.
Look for: "read", "R:", "read/write" (first number), "MB/s read".
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: CORSAIR Force Series MP510 960GB M.2 SSD, up to 3480/3000MB/s read/write
VALUE:3480

Example 2:
Product: Samsung 860 EVO 4TB 2.5" SATA SSD. Description: Read 550MB/s, Write 520MB/s
VALUE:550

Example 3:
Product: SSD 240GB 2.5'' PATRIOT Burst SATA3 R/W:555/500 MB/s 3D NAND
VALUE:555

Now extract:
Product: {text}
VALUE:""",

"write_speed_mb_s": """\
You are a product data expert. Extract ONLY the sequential WRITE speed in MB/s from the product text.
Return the number only (e.g. 3000, 520, 2500). Do NOT return read speed.
Look for: "write", "W:", "read/write" (second number), "MB/s write".
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: CORSAIR Force Series MP510 960GB M.2 SSD, up to 3480/3000MB/s read/write
VALUE:3000

Example 2:
Product: Samsung 860 EVO 4TB 2.5" SATA SSD. Description: Read 550MB/s, Write 520MB/s
VALUE:520

Example 3:
Product: SSD 240GB 2.5'' PATRIOT Burst SATA3 R/W:555/500 MB/s 3D NAND
VALUE:500

Now extract:
Product: {text}
VALUE:""",

"height_mm": """\
You are a product data expert. Extract ONLY the HEIGHT in millimeters from the product text.
Return the number only (e.g. 46.0, 10.5, 7.0).
Look for: "height", "H:", dimensions format HxWxD, "mm" measurements.
Do NOT confuse with width or length.
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: GeForce GTX 1660 Ti GAMING X 6G. Description: Dimensions: 46mm x 127mm x 247mm (H x W x D)
VALUE:46

Example 2:
Product: Samsung T5 1TB External SSD. Description: Dimensions (HDW): 10.5mm x 74mm x 57mm
VALUE:10.5

Example 3:
Product: Kingston DC500M 1.92TB 2.5" SSD. Description: SATA 3.0, 2.5", up to 555MB/s read, 7mm height
VALUE:7

Now extract:
Product: {text}
VALUE:""",

"width_mm": """\
You are a product data expert. Extract ONLY the WIDTH in millimeters from the product text.
Return the number only (e.g. 127.0, 57.0, 22.0).
Look for: "width", "W:", dimensions format HxWxD, "mm" measurements.
Do NOT confuse with height or length.
Respond with VALUE:<number> only. If not found, respond with VALUE:UNKNOWN.

Example 1:
Product: GeForce GTX 1660 Ti GAMING X 6G. Description: Dimensions: 46mm x 127mm x 247mm (H x W x D)
VALUE:127

Example 2:
Product: Samsung T5 1TB External SSD. Description: Dimensions (HDW): 10.5mm x 74mm x 57mm
VALUE:57

Example 3:
Product: ADATA SU800 SATA M.2 2280 SSD 256GB. Description: 22x80x3.5mm dimensions
VALUE:22

Now extract:
Product: {text}
VALUE:"""
}

print("LLM-only prompts defined for:", list(FEW_SHOT_PROMPTS.keys()))


# ## 10. RAG Prompts (with KB candidates)

# In[26]:


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

print("RAG prompts defined for:", list(FEW_SHOT_PROMPTS_RAG.keys()))


# ## 11. Format Candidates

# In[27]:


SKIP_FIELDS = {
    "id", "url", "description", "title_description",
    "price", "priceCurrency", "cluster_id"
}

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

print("format_candidates() ready.")


# ## 12. Run Prediction

# In[ ]:





# In[ ]:





# In[29]:


def predict_attribute(product_text, attribute, predict_model, candidates=None):
    if candidates is not None:
        cands_text = format_candidates(candidates)
        prompt_template = FEW_SHOT_PROMPTS_RAG[attribute]
        prompt = prompt_template.format(
            text=str(product_text)[:500],
            candidates=cands_text
        )
    else:
        prompt_template = FEW_SHOT_PROMPTS[attribute]
        prompt = prompt_template.format(text=str(product_text)[:600])

    response = predict_model.invoke([HumanMessage(content=prompt)])
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


def predict_with_timeout(product_text, attribute, predict_model, candidates=None, timeout=300):
    result = ["UNKNOWN"]
    def target():
        try:
            result[0] = predict_attribute(product_text, attribute, predict_model, candidates)
        except:
            result[0] = "UNKNOWN"
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        print(f"    ⚠️ TIMEOUT — skipping")
        return "UNKNOWN"
    return result[0]


print(f"Running prediction (RAG={'ON' if USE_RAG else 'OFF'}, "
      f"retrieval: MiniLM top-{TOP_N} → CrossEncoder top-{TOP_K})...")
t0 = time.time()
total = len(eval_df)
predictions = []

for i, (_, task) in enumerate(eval_df.iterrows()):
    idx = task["df1_idx"]
    attr = task["attribute"]
    gt = task["ground_truth"]
    text = query_df.loc[idx, "title_description"]

    if USE_RAG:
        pos = query_idx_to_pos[idx]
        q_emb = query_embeddings[pos]
        query_text = query_texts[pos]
        candidates, _ = retrieve_and_rerank(query_text, q_emb)
    else:
        candidates = None

    predicted = predict_with_timeout(text, attr, predict_model, candidates, timeout=300)

    elapsed = time.time() - t0
    eta = (elapsed / (i+1)) * (total - i - 1)
    print(f"  [{i+1}/{total}] Row {idx} | {attr:<20} | "
          f"GT: {str(gt):<25} | Pred: {predicted:<25} | ETA: {eta/60:.1f}min")

    predictions.append({
        "df1_idx": idx,
        "config": "RAG" if USE_RAG else "LLM-only",
        "attribute": attr,
        "is_numeric": task["is_numeric"],
        "ground_truth": gt,
        "predicted": predicted,
        "unknown": predicted == "UNKNOWN"
    })

results_df = pd.DataFrame(predictions)
print(f"\nDone in {time.time()-t0:.1f}s")


# ## 13. Retrieval Hit@k Analysis (before and after re-ranking)

# In[ ]:


if USE_RAG:
    print(f"=== RETRIEVAL QUALITY: MiniLM Hit@{TOP_N} vs CrossEncoder Hit@{TOP_K} ===\n")

    hit_minilm = 0
    hit_crossenc = 0
    total_tasks = 0

    for _, task in eval_df.iterrows():
        idx = task["df1_idx"]
        cluster_id = query_df.loc[idx, "cluster_id"]

        pos = query_idx_to_pos[idx]
        q_emb = query_embeddings[pos]
        query_text = query_texts[pos]

        # MiniLM top-N
        scores = util.cos_sim(q_emb, kb_embeddings)[0]
        top_n_idx = np.argsort(-scores.cpu().numpy())[:TOP_N]
        candidates_n = kb.iloc[top_n_idx]
        minilm_hit = cluster_id in candidates_n["cluster_id"].values

        # CrossEncoder top-K
        candidates_k, _ = retrieve_and_rerank(query_text, q_emb)
        crossenc_hit = cluster_id in candidates_k["cluster_id"].values

        hit_minilm += int(minilm_hit)
        hit_crossenc += int(crossenc_hit)
        total_tasks += 1

        minilm_sym = "✓" if minilm_hit else "✗"
        crossenc_sym = "✓" if crossenc_hit else "✗"
        print(f"  Row {idx:<4} | cluster {cluster_id} | "
              f"MiniLM@{TOP_N}: {minilm_sym} | CrossEnc@{TOP_K}: {crossenc_sym}")

    print(f"\nMiniLM Hit@{TOP_N}:      {hit_minilm}/{total_tasks} = {hit_minilm/total_tasks:.3f}")
    print(f"CrossEncoder Hit@{TOP_K}: {hit_crossenc}/{total_tasks} = {hit_crossenc/total_tasks:.3f}")
    print(f"Re-ranking improvement:   {(hit_crossenc-hit_minilm):+d} tasks")
else:
    print("Skipping Hit@k — LLM-only mode, no retrieval.")


# ## 14. Standard Evaluation

# In[ ]:


def normalize(val):
    return str(val).lower().strip()

def is_correct_standard(predicted, ground_truth, attribute):
    if predicted is None or str(predicted).strip().lower() in {
            "", "nan", "none", "unknown", "null"}:
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(",", "").strip())
            g = float(str(ground_truth).replace(",", "").strip())
            return abs(p - g) / abs(g) <= 0.10 if g != 0 else p == 0
        except:
            pass
    p = normalize(str(predicted))
    g = normalize(str(ground_truth))
    return p == g or p in g or g in p

results_df["correct_standard"] = results_df.apply(
    lambda row: is_correct_standard(row["predicted"], row["ground_truth"], row["attribute"]),
    axis=1
)

print("=" * 55)
print("STANDARD EVALUATION")
print("=" * 55)
print(f"Overall accuracy: {results_df['correct_standard'].mean():.3f}")
print(f"UNKNOWN rate:     {results_df['unknown'].mean():.3f}")
print(f"Total tasks:      {len(results_df)}")
print()
print("Per-attribute:")
print(results_df.groupby("attribute").agg(
    total=("correct_standard", "count"),
    correct=("correct_standard", "sum"),
    accuracy=("correct_standard", "mean"),
    unknown_rate=("unknown", "mean")
).round(3).to_string())


# ## 15. Improved Evaluation
# 
# **Key methodological improvement over experiments 8.1 and 9:**
# - Numeric attributes: pure rule-based (10% tolerance) — no LLM needed
# - Text attributes: CrossEncoder similarity score — removes same-model bias
# - Llama is used only for prediction, not evaluation

# In[ ]:


def evaluate_prediction(predicted, ground_truth, attribute):
    """Improved evaluation: rule-based for numeric, CrossEncoder for text."""
    if predicted == "UNKNOWN" or str(predicted).lower() in {"nan", "none", "null", ""}:
        return "wrong"

    if attribute in NUMERIC_ATTRIBUTES:
        # Pure rule-based — no model needed
        try:
            p = float(str(predicted).replace(",", "").strip())
            g = float(str(ground_truth).replace(",", "").strip())
            if g == 0:
                return "correct" if p == 0 else "wrong"
            ratio = abs(p - g) / abs(g)
            if ratio <= 0.10:
                return "correct"
            elif ratio <= 0.30:
                return "acceptable"
            else:
                return "wrong"
        except:
            return "wrong"

    else:
        # CrossEncoder similarity for text attributes
        # Score range roughly -10 to +10; we calibrate thresholds empirically
        score = cross_encoder.predict([[ground_truth, predicted]])[0]
        if score > 2.0:
            return "correct"
        elif score > -1.0:
            return "acceptable"
        else:
            return "wrong"


print("Running improved evaluation (CrossEncoder for text, rule-based for numeric)...")
t0 = time.time()
judgments = []

for i, (_, row) in enumerate(results_df.iterrows()):
    judgment = evaluate_prediction(
        row["predicted"], row["ground_truth"], row["attribute"]
    )
    judgments.append(judgment)
    print(f"  [{i+1}/{len(results_df)}] {row['attribute']:<20} | "
          f"GT: {str(row['ground_truth']):<25} | "
          f"Pred: {str(row['predicted']):<25} | {judgment}")

results_df["ce_judgment"] = judgments
results_df.to_csv(RESULT_FILE, index=False)
print(f"\nDone in {time.time()-t0:.1f}s — saved to {RESULT_FILE}")


# ## 16. Final Summary

# In[ ]:


standard_acc = results_df["correct_standard"].mean()
ce_correct = (results_df["ce_judgment"] == "correct").mean()
ce_acceptable = (results_df["ce_judgment"].isin(["correct", "acceptable"])).mean()

print("=" * 60)
print(f"RESULTS — {'RAG + CrossEncoder Re-ranking' if USE_RAG else 'LLM-only'} (exp 10)")
print("=" * 60)
print(f"{'Standard accuracy':<45} {standard_acc:.3f}")
print(f"{'CE eval — correct only':<45} {ce_correct:.3f}")
print(f"{'CE eval — correct + acceptable':<45} {ce_acceptable:.3f}")
print(f"{'UNKNOWN rate':<45} {results_df['unknown'].mean():.3f}")

print("\nPer-attribute breakdown:")
per_attr = results_df.groupby("attribute").apply(lambda x: pd.Series({
    "total": len(x),
    "standard_acc": x["correct_standard"].mean(),
    "ce_correct+acceptable": (x["ce_judgment"].isin(["correct", "acceptable"])).mean(),
    "unknown_rate": x["unknown"].mean()
}), include_groups=False).round(3)
print(per_attr.to_string())

print("\nCE judgment distribution:")
print(results_df["ce_judgment"].value_counts())


# ## 17. Combined Comparison with Experiments 8.1 and 9

# In[ ]:


import os
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.exists(LLM_FILE) and os.path.exists(RAG_FILE):
    results_llm = pd.read_csv(LLM_FILE)
    results_rag = pd.read_csv(RAG_FILE)
    results_llm["config"] = "LLM-only"
    results_rag["config"] = "RAG+Reranker"
    all_exp10 = pd.concat([results_llm, results_rag], ignore_index=True)
    all_exp10.to_csv(FINAL_RESULTS_FILE, index=False)

    # Optionally load exp8.1 for comparison
    configs_to_compare = []
    if os.path.exists("results_exp8_1_all.csv"):
        exp81 = pd.read_csv("results_exp8_1_all.csv")
        exp81["config"] = exp81["config"].replace({"RAG": "RAG (exp8.1)", "LLM-only": "LLM-only"})
        configs_to_compare.append(exp81)

    configs_to_compare.append(all_exp10)
    combined = pd.concat(configs_to_compare, ignore_index=True)

    print("=" * 70)
    print("COMPARISON: exp 8.1 vs exp 10 (with CrossEncoder re-ranking)")
    print("=" * 70)

    for config in combined["config"].unique():
        df_c = combined[combined["config"] == config]
        print(f"\n--- {config} ---")
        print(f"  Standard accuracy: {df_c['correct_standard'].mean():.3f}")
        print(f"  UNKNOWN rate:      {df_c['unknown'].mean():.3f}")
        if "ce_judgment" in df_c.columns:
            ce = (df_c["ce_judgment"].isin(["correct","acceptable"])).mean()
            print(f"  CE eval:           {ce:.3f}")
        elif "llm_judgment" in df_c.columns:
            llm = (df_c["llm_judgment"].isin(["correct","acceptable"])).mean()
            print(f"  LLM eval:          {llm:.3f}")

    # Heatmap — standard accuracy
    heatmap_data = (
        combined.groupby(["attribute", "config"])["correct_standard"]
        .mean().unstack("config")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        cmap="Blues", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Standard Accuracy"}
    )
    ax.set_title("Experiment 10 — Standard Accuracy: CrossEncoder Re-ranking vs Baseline",
                 fontsize=12, pad=10)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Attribute")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("fig_exp10_easy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓ Saved to fig_exp10_easy_heatmap.png")
    print(f"✓ Saved to {FINAL_RESULTS_FILE}")
else:
    print(f"Run with USE_RAG=False first to generate {LLM_FILE}, then rerun with USE_RAG=True.")


# In[ ]:


# See what the reranker is actually retrieving for the failing rows
for fail_idx in [33, 41, 52, 77]:
    if fail_idx not in query_idx_to_pos:
        print(f"Row {fail_idx} not in query set")
        continue

    pos = query_idx_to_pos[fail_idx]
    q_emb = query_embeddings[pos]
    query_text = query_texts[pos]

    candidates, scores = retrieve_and_rerank(query_text, q_emb)

    true_cluster = query_df.loc[fail_idx, "cluster_id"]
    print(f"\nRow {fail_idx} | cluster {true_cluster} | query: {query_text[:60]}")
    for i, (_, c) in enumerate(candidates.iterrows()):
        match = "✓" if c["cluster_id"] == true_cluster else "✗"
        print(f"  [{i+1}] {match} score={scores[i]:.2f} | cluster {c['cluster_id']} | {str(c.get('title',''))[:60]}")      


# In[ ]:




