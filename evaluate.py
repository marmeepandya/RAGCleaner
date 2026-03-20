import sys
sys.path.append("/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/PyDI")

import random
import time
import numpy as np
import requests
import pandas as pd
from sentence_transformers import util
from cleaners.rag_cleaner import RAGCleaner

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
df2 = pd.read_json("normalized_products/dataset_2_normalized.json")
df3 = pd.read_json("normalized_products/dataset_3_normalized.json")
df4 = pd.read_json("normalized_products/dataset_4_normalized.json")

kb = pd.concat([df2, df3, df4], ignore_index=True)
print(f"Knowledge base size: {len(kb)}")

# ── Build eval set using cluster_id ───────────────────────────────────────────
print("Building evaluation set...")

target_attributes = [
    "model_number", "storage_size", "bus_type",
    "interface_type", "form_factor", "vram", "storage_connection_type"
]

def get_ground_truth(cluster_id, attribute, kb):
    matches = kb[kb["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

eval_records = []
for idx, row in df1.iterrows():
    for attr in target_attributes:
        if pd.isna(row.get(attr)):
            gt = get_ground_truth(row["cluster_id"], attr, kb)
            if gt is not None:
                eval_records.append({
                    "df1_idx": idx,
                    "cluster_id": row["cluster_id"],
                    "attribute": attr,
                    "ground_truth": gt
                })

eval_df = pd.DataFrame(eval_records)
print(f"Eval set size: {len(eval_df)}")
print(eval_df["attribute"].value_counts())

# Drop attributes with too few samples
eval_df = eval_df[~eval_df["attribute"].isin(["storage_size", "vram"])]
print(f"Filtered eval set: {len(eval_df)} tasks")

# ── LLM Wrapper ───────────────────────────────────────────────────────────────
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

# ── LLM-only cleaner ──────────────────────────────────────────────────────────
class LLMOnlyCleaner:
    def __init__(self, llm):
        self.llm = llm

    def clean_cell(self, row, attribute):
        product = {k: v for k, v in row.items()
                   if k in ["title", "description", "model", "brand", "product_type"]
                   and pd.notna(v)}

        # Clearer prompt with examples to force format
        prompt = f"""You are a product data expert. Fill in the missing attribute for this product.

PRODUCT: {product}

MISSING ATTRIBUTE: {attribute}

You must respond with ONLY this exact format, nothing else:
VALUE:<your answer>

Example response: VALUE:PCIe
Example response: VALUE:M.2

Your response:"""

        response = self.llm.generate(prompt)
        return self._parse_response(response)

    _EMPTY_SENTINELS = {"none", "nan", "null", "n/a", "na", "unknown", ""}

    def _parse_response(self, response: str) -> str:
        # Try VALUE: format first
        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("VALUE:"):
                value = line.split(":", 1)[1].strip().strip('"').strip("'")
                if value.lower() not in self._EMPTY_SENTINELS:
                    return value

        # Fallback: if model returns a short raw answer, use it directly
        cleaned = response.strip().strip('"').strip("'")
        if cleaned and "\n" not in cleaned and len(cleaned) < 50:
            if cleaned.lower() not in self._EMPTY_SENTINELS:
                return cleaned

        return "UNKNOWN"

# ── Evaluation runner ─────────────────────────────────────────────────────────
def normalize(val):
    return str(val).lower().strip()

def is_correct(predicted, ground_truth):
    p = normalize(predicted)
    g = normalize(ground_truth)
    # exact match
    if p == g:
        return True
    # partial match — predicted is contained in gt or vice versa
    if p in g or g in p:
        return True
    return False

def run_evaluation(eval_df, df1, cleaner, config_name, save_path=None):
    results = []
    total = len(eval_df)
    rows = [df1.loc[task["df1_idx"]] for _, task in eval_df.iterrows()]

    query_embeddings = None
    if hasattr(cleaner, 'kb_embeddings'):
        print(f"Precomputing query embeddings for {config_name}...")
        query_texts = [cleaner._row_to_text(row) for row in rows]
        query_embeddings = cleaner.model.encode(
            query_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=True
        )
        print("Done.")

    config_start = time.time()

    for i, (_, task) in enumerate(eval_df.iterrows()):
        t0 = time.time()
        row = rows[i]

        if query_embeddings is not None:
            cos_scores = util.cos_sim(query_embeddings[i], cleaner.kb_embeddings)[0]
            top_indices = np.argsort(-cos_scores.cpu().numpy())[:cleaner.top_k]
            candidates = cleaner.kb.iloc[top_indices]
            prompt = cleaner._build_prompt(row, candidates, task["attribute"])
            response = cleaner.llm.generate(prompt)
            predicted = cleaner._parse_response(response)
        else:
            predicted = cleaner.clean_cell(row, task["attribute"])

        gt = task["ground_truth"]
        correct = is_correct(predicted, gt)
        elapsed = time.time() - t0

        results.append({
            "config": config_name,
            "attribute": task["attribute"],
            "ground_truth": gt,
            "predicted": predicted,
            "correct": correct,
            "unknown": predicted == "UNKNOWN"
        })

        if i % 10 == 0:
            elapsed_total = time.time() - config_start
            remaining = (elapsed_total / (i + 1)) * (total - i - 1)
            print(f"[{config_name}] {i+1}/{total} | last task: {elapsed:.2f}s | "
                  f"ETA: {remaining/60:.1f} min | attr: {task['attribute']}")

    results_df = pd.DataFrame(results)
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"✓ Saved {config_name} results to {save_path}")
    return results_df

# ── Run all configs on bus_type only ─────────────────────────────────────────
llm = OllamaLLMWrapper("llama3.1:8b")
eval_df_bus = eval_df[eval_df["attribute"] == "model_number"]

# Config 1: LLM-only
print("\n=== CONFIG 1: LLM-only ===")
results_llm = run_evaluation(
    eval_df_bus, df1, LLMOnlyCleaner(llm), "LLM-only",
    save_path="results_bus_llm_only.csv"
)
print(f"Accuracy: {results_llm['correct'].mean():.3f}")

# Config 2: RAG full KB
print("\n=== CONFIG 2: RAG full KB ===")
rag_full = RAGCleaner(knowledge_base=kb, llm=llm, top_k=3)
results_rag_full = run_evaluation(
    eval_df_bus, df1, rag_full, "RAG-full",
    save_path="results_bus_rag_full.csv"
)
print(f"Accuracy: {results_rag_full['correct'].mean():.3f}")

# Config 3: RAG partial KB (50%)
print("\n=== CONFIG 3: RAG partial KB ===")
kb_partial = kb.sample(frac=0.5, random_state=42).reset_index(drop=True)
rag_partial = RAGCleaner(knowledge_base=kb_partial, llm=llm, top_k=3)
results_rag_partial = run_evaluation(
    eval_df_bus, df1, rag_partial, "RAG-partial",
    save_path="results_bus_rag_partial.csv"
)
print(f"Accuracy: {results_rag_partial['correct'].mean():.3f}")

# Summary
print("\n=== BUS_TYPE SUMMARY ===")
for results, name in [(results_llm, "LLM-only"), (results_rag_full, "RAG-full"), (results_rag_partial, "RAG-partial")]:
    print(f"{name:15} | Accuracy: {results['correct'].mean():.3f} | UNKNOWN rate: {results['unknown'].mean():.3f}")