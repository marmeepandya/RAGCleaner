import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import requests
import time
import random

random.seed(42)
np.random.seed(42)

LOCAL_MODEL_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

TARGET_ATTRIBUTES = [
    "bus_type", "model_number", "model",
    "read_speed_mb_s", "write_speed_mb_s",
    "height_mm", "width_mm"
]
NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
df2 = pd.read_json("normalized_products/dataset_2_normalized.json")
df3 = pd.read_json("normalized_products/dataset_3_normalized.json")
df4 = pd.read_json("normalized_products/dataset_4_normalized.json")
kb_clean = pd.concat([df2, df3, df4], ignore_index=True)
print(f"Dataset 1: {len(df1)} rows, KB: {len(kb_clean)} rows")

# ── Build eval set ─────────────────────────────────────────────────────────────
def get_ground_truth(cluster_id, attribute, kb):
    matches = kb[kb["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

eval_records = []
for idx, row in df1.iterrows():
    for attr in TARGET_ATTRIBUTES:
        if pd.isna(row.get(attr)):
            gt = get_ground_truth(row["cluster_id"], attr, kb_clean)
            if gt is not None:
                eval_records.append({
                    "df1_idx": idx,
                    "cluster_id": row["cluster_id"],
                    "attribute": attr,
                    "ground_truth": gt,
                    "is_numeric": attr in NUMERIC_ATTRIBUTES
                })

eval_df = pd.DataFrame(eval_records)
print(f"Full eval set: {len(eval_df)} tasks")
print(eval_df["attribute"].value_counts())

# Sample 25 unique rows
unique_rows = eval_df["df1_idx"].unique()
sampled_rows = pd.Series(unique_rows).sample(n=min(25, len(unique_rows)), random_state=42).values
eval_df = eval_df[eval_df["df1_idx"].isin(sampled_rows)].reset_index(drop=True)
print(f"\nSampled eval set: {len(eval_df)} tasks across 25 rows")
print(eval_df["attribute"].value_counts())

# ── Noisy KB ───────────────────────────────────────────────────────────────────
def build_noisy_kb(clean_kb, target_attributes, noise_fraction=0.5, random_state=42):
    noisy_kb = clean_kb.copy()
    rng = np.random.default_rng(random_state)
    for attr in target_attributes:
        if attr not in noisy_kb.columns:
            continue
        non_null_idx = noisy_kb[attr].dropna().index.tolist()
        if len(non_null_idx) == 0:
            continue
        n_to_corrupt = int(len(non_null_idx) * noise_fraction)
        corrupt_idx = rng.choice(non_null_idx, size=n_to_corrupt, replace=False)
        all_values = noisy_kb[attr].dropna().tolist()
        random_values = rng.choice(all_values, size=n_to_corrupt)
        noisy_kb.loc[corrupt_idx, attr] = random_values
    return noisy_kb

kb_noisy = build_noisy_kb(kb_clean, TARGET_ATTRIBUTES)
print("Noisy KB built.")

# ── Embeddings ─────────────────────────────────────────────────────────────────
print("Loading embedding model...")
embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)

def row_to_text(row):
    attrs = ["title", "model", "model_number", "brand", "product_type"]
    return " | ".join([str(row[a]) for a in attrs if pd.notna(row.get(a))])

def encode_kb(kb):
    kb = kb.copy().reset_index(drop=True)
    kb["_text"] = kb.apply(row_to_text, axis=1)
    embeddings = embedding_model.encode(
        kb["_text"].tolist(), convert_to_tensor=True,
        batch_size=64, show_progress_bar=True
    )
    return kb, embeddings

print("Encoding clean KB...")
kb_clean_enc, kb_clean_emb = encode_kb(kb_clean)
print("Encoding noisy KB...")
kb_noisy_enc, kb_noisy_emb = encode_kb(kb_noisy)

# Precompute query embeddings
print("Encoding query rows...")
rows = [df1.loc[task["df1_idx"]] for _, task in eval_df.iterrows()]
query_texts = [row_to_text(r) for r in rows]
query_embeddings = embedding_model.encode(
    query_texts, convert_to_tensor=True,
    batch_size=64, show_progress_bar=True
)

# ── LLM ────────────────────────────────────────────────────────────────────────
class OllamaLLMWrapper:
    def __init__(self, model_name="llama3.1:8b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

llm = OllamaLLMWrapper("llama3.1:8b")
print("LLM OK:", repr(llm.generate("Say OK")[:20]))

# ── Prompts ────────────────────────────────────────────────────────────────────
_EMPTY = {"none", "nan", "null", "n/a", "na", "unknown", ""}

def parse_response(response):
    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("VALUE:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if value.lower() not in _EMPTY:
                return value
    cleaned = response.strip().strip('"').strip("'")
    if cleaned and "\n" not in cleaned and len(cleaned) < 80:
        if cleaned.lower() not in _EMPTY:
            return cleaned
    return "UNKNOWN"

def build_prompt_llm_only(row, attribute):
    product = {k: v for k, v in row.items()
               if k in ["title", "model", "brand", "product_type"] and pd.notna(v)}
    example = "VALUE:3500" if attribute in NUMERIC_ATTRIBUTES else "VALUE:PCIe"
    type_hint = " (numeric)" if attribute in NUMERIC_ATTRIBUTES else ""
    return (
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"MISSING ATTRIBUTE: {attribute}{type_hint}\n\n"
        f"Respond with VALUE:<answer> only. Example: {example}"
    )

def build_prompt_rag(row, attribute, candidates):
    product = {k: v for k, v in row.items()
               if k in ["title", "model", "brand", "product_type"] and pd.notna(v)}
    key_fields = ["title", "model", "model_number", "bus_type",
                  "read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"]
    cands_lines = []
    for _, c in candidates.iterrows():
        c_dict = {k: v for k, v in c.items()
                  if k in key_fields and pd.notna(c.get(k))}
        cands_lines.append(str(c_dict))
    cands_text = "\n".join(cands_lines)
    example = "VALUE:3500" if attribute in NUMERIC_ATTRIBUTES else "VALUE:PCIe"
    type_hint = " (numeric)" if attribute in NUMERIC_ATTRIBUTES else ""
    return (
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"SIMILAR REFERENCE PRODUCTS:\n{cands_text}\n\n"
        f"MISSING ATTRIBUTE: {attribute}{type_hint}\n\n"
        f"Use the reference products to find the correct value. "
        f"Respond with VALUE:<answer> only. Example: {example}"
    )

# ── Scoring ────────────────────────────────────────────────────────────────────
def normalize(val):
    return str(val).lower().strip()

def is_correct(predicted, ground_truth, attribute):
    if predicted == "UNKNOWN":
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(",", "").strip())
            g = float(str(ground_truth).replace(",", "").strip())
            return abs(p - g) / abs(g) <= 0.10 if g != 0 else p == 0
        except:
            pass
    p, g = normalize(predicted), normalize(ground_truth)
    return p == g or p in g or g in p

def llm_evaluate(predicted, ground_truth, attribute):
    if predicted == "UNKNOWN":
        return "wrong"
    prompt = (
        f"You are evaluating a data cleaning prediction.\n\n"
        f"Attribute: {attribute}\n"
        f"Ground truth: {ground_truth}\n"
        f"Predicted: {predicted}\n\n"
        f"Judge as one of:\n"
        f"- CORRECT: exact match or equivalent\n"
        f"- ACCEPTABLE: more detailed but not wrong\n"
        f"- WRONG: incorrect\n\n"
        f"Respond with JUDGMENT:<label> only. Example: JUDGMENT:CORRECT"
    )
    response = llm.generate(prompt)
    for line in response.splitlines():
        line = line.strip().upper()
        if line.startswith("JUDGMENT:"):
            label = line.split(":", 1)[1].strip()
            if label in {"CORRECT", "ACCEPTABLE", "WRONG"}:
                return label.lower()
    r = response.upper()
    if "CORRECT" in r: return "correct"
    if "ACCEPTABLE" in r: return "acceptable"
    return "wrong"

# ── Evaluation runner ──────────────────────────────────────────────────────────
def run_evaluation(eval_df, rows, query_embeddings, kb_enc, kb_emb,
                   config_name, use_rag, top_k=3,
                   use_llm_eval=False, save_path=None):
    results = []
    total = len(eval_df)
    config_start = time.time()

    for i, (_, task) in enumerate(eval_df.iterrows()):
        t0 = time.time()
        row = rows[i]
        attr = task["attribute"]
        gt = task["ground_truth"]

        if use_rag:
            scores = util.cos_sim(query_embeddings[i], kb_emb)[0]
            top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
            candidates = kb_enc.iloc[top_idx]
            prompt = build_prompt_rag(row, attr, candidates)
        else:
            prompt = build_prompt_llm_only(row, attr)

        predicted = parse_response(llm.generate(prompt))
        correct = is_correct(predicted, gt, attr)
        llm_judgment = llm_evaluate(predicted, gt, attr) if use_llm_eval else None

        results.append({
            "config": config_name,
            "attribute": attr,
            "is_numeric": task["is_numeric"],
            "ground_truth": gt,
            "predicted": predicted,
            "correct": correct,
            "llm_judgment": llm_judgment,
            "unknown": predicted == "UNKNOWN"
        })

        elapsed = time.time() - config_start
        remaining = (elapsed / (i+1)) * (total - i - 1)
        print(f"[{config_name}] {i+1}/{total} | "
              f"{time.time()-t0:.2f}s | "
              f"ETA: {remaining/60:.1f} min | "
              f"{attr} | pred: {predicted} | gt: {gt} | {'✓' if correct else '✗'}")

    results_df = pd.DataFrame(results)
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"✓ Saved [{config_name}] → {save_path}")
    return results_df

# ── Run all configs ────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("CONFIG 1: LLM-only")
print("="*50)
results_llm = run_evaluation(
    eval_df, rows, query_embeddings,
    kb_clean_enc, kb_clean_emb,
    "LLM-only", use_rag=False,
    use_llm_eval=True,
    save_path="results_exp4_llm_only.csv"
)

print("\n" + "="*50)
print("CONFIG 2: RAG clean KB k=3")
print("="*50)
results_rag_clean = run_evaluation(
    eval_df, rows, query_embeddings,
    kb_clean_enc, kb_clean_emb,
    "RAG-clean-k3", use_rag=True, top_k=3,
    use_llm_eval=True,
    save_path="results_exp4_rag_clean_k3.csv"
)

print("\n" + "="*50)
print("CONFIG 3: RAG noisy KB k=3")
print("="*50)
results_rag_noisy = run_evaluation(
    eval_df, rows, query_embeddings,
    kb_noisy_enc, kb_noisy_emb,
    "RAG-noisy-k3", use_rag=True, top_k=3,
    use_llm_eval=True,
    save_path="results_exp4_rag_noisy_k3.csv"
)

print("\n" + "="*50)
print("CONFIG 4: RAG clean KB k=5")
print("="*50)
results_rag_clean_k5 = run_evaluation(
    eval_df, rows, query_embeddings,
    kb_clean_enc, kb_clean_emb,
    "RAG-clean-k5", use_rag=True, top_k=5,
    use_llm_eval=True,
    save_path="results_exp4_rag_clean_k5.csv"
)

# ── Summary ────────────────────────────────────────────────────────────────────
import pandas as pd
all_results = pd.concat([
    results_llm, results_rag_clean,
    results_rag_noisy, results_rag_clean_k5
], ignore_index=True)
all_results.to_csv("results_exp4_all.csv", index=False)

print("\n=== OVERALL ACCURACY ===")
print(all_results.groupby("config")["correct"].mean().round(3))

print("\n=== PER-ATTRIBUTE ACCURACY ===")
print(all_results.groupby(["config", "attribute"])["correct"].mean().round(3).unstack())

print("\n=== LLM EVALUATION ===")
llm_eval = all_results[all_results["llm_judgment"].notna()]
print(llm_eval.groupby(["config", "llm_judgment"]).size().unstack(fill_value=0))

print("\n=== UNKNOWN RATE ===")
print(all_results.groupby("config")["unknown"].mean().round(3))

print("\nDone!")
