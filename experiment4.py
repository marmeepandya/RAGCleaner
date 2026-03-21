#!/usr/bin/env python
# coding: utf-8

# # Experiment 4 — Redesigned RAG Evaluation
# 
# Key changes from experiment 3:
# - **All 4 datasets** used as query (rotating KB)
# - **Clean KB** vs **Noisy KB** (50% correct + 50% random injected values)
# - **New attributes** including numeric ones: `read_speed_mb_s`, `write_speed_mb_s`, `height_mm`, `width_mm`
# - **LLM-based evaluation** in addition to partial match
# - **Prompt variants** tested

# In[1]:


import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LOCAL_MODEL_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)
print("Model loaded successfully!")


# ## 1. Imports & Setup

# In[2]:


import pandas as pd
import numpy as np
import requests
import time
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

sns.set_theme(style="whitegrid", palette="muted")
random.seed(42)
np.random.seed(42)

TARGET_ATTRIBUTES = [
    "bus_type",        # text, easy — LLM baseline
    "model_number",    # text, hard — needs retrieval
    "model",           # text, medium
    "read_speed_mb_s", # numeric
    "write_speed_mb_s",# numeric
    "height_mm",       # numeric
    "width_mm",        # numeric
]

NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}


# ## 2. Load All Datasets

# In[19]:


dfs = {
    1: pd.read_json("normalized_products/dataset_1_normalized.json"),
    2: pd.read_json("normalized_products/dataset_2_normalized.json"),
    3: pd.read_json("normalized_products/dataset_3_normalized.json"),
    4: pd.read_json("normalized_products/dataset_4_normalized.json"),
}
for i, df in dfs.items():
    print(f"Dataset {i}: {len(df)} rows")


# ## 3. Build Evaluation Set (All 4 Query Datasets)

# In[20]:


def get_ground_truth(cluster_id, attribute, kb):
    matches = kb[kb["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

all_eval_records = []

for query_idx in [1, 2, 3, 4]:
    df_query = dfs[query_idx]
    kb_clean = pd.concat([dfs[i] for i in [1,2,3,4] if i != query_idx],
                          ignore_index=True)
    for idx, row in df_query.iterrows():
        for attr in TARGET_ATTRIBUTES:
            if pd.isna(row.get(attr)):
                gt = get_ground_truth(row["cluster_id"], attr, kb_clean)
                if gt is not None:
                    all_eval_records.append({
                        "query_dataset": query_idx,
                        "df_idx": idx,
                        "cluster_id": row["cluster_id"],
                        "attribute": attr,
                        "ground_truth": gt,
                        "is_numeric": attr in NUMERIC_ATTRIBUTES
                    })

eval_df = pd.DataFrame(all_eval_records)
print(f"Total eval tasks: {len(eval_df)}")
print(eval_df.groupby(["query_dataset", "attribute"]).size().unstack(fill_value=0))


# ## 4. Build Knowledge Bases
# ### 4a. Clean KB (correct values only)

# In[21]:


# Clean KB per query dataset (the other 3 datasets, unmodified)
clean_kbs = {
    q: pd.concat([dfs[i] for i in [1,2,3,4] if i != q], ignore_index=True)
    for q in [1, 2, 3, 4]
}
print("Clean KB sizes:", {q: len(kb) for q, kb in clean_kbs.items()})


# ### 4b. Noisy KB (50% correct + 50% random injected values)

# In[22]:


def build_noisy_kb(clean_kb, target_attributes, noise_fraction=0.5, random_state=42):
    """
    Build a noisy KB by replacing noise_fraction of attribute values
    with random values sampled from the same column.
    This simulates a KB with plausible but incorrect values.
    """
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
        # Sample random values from the same column (plausible but wrong)
        all_values = noisy_kb[attr].dropna().tolist()
        random_values = rng.choice(all_values, size=n_to_corrupt)
        noisy_kb.loc[corrupt_idx, attr] = random_values

    return noisy_kb

noisy_kbs = {
    q: build_noisy_kb(clean_kbs[q], TARGET_ATTRIBUTES)
    for q in [1, 2, 3, 4]
}
print("Noisy KBs built. Sample corruption check:")
for attr in ["bus_type", "model_number", "read_speed_mb_s"]:
    clean_sample = clean_kbs[1][attr].dropna().iloc[:3].tolist()
    noisy_sample = noisy_kbs[1][attr].dropna().iloc[:3].tolist()
    print(f"  {attr}: clean={clean_sample} | noisy={noisy_sample}")


# ## 5. LLM & Cleaner Setup

# In[23]:


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
print("Ollama OK:", repr(llm.generate("Say OK")[:20]))


# ## 6. Prompt Variants

# In[24]:


def build_prompt_basic(row, attribute, candidates=None):
    """Prompt A: minimal, just product info."""
    product = {k: v for k, v in row.items()
               if k in ["title", "model", "brand", "product_type"] and pd.notna(v)}
    prompt = (
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"MISSING ATTRIBUTE: {attribute}\n\n"
        f"Respond with VALUE:<answer> only. Example: VALUE:PCIe"
    )
    return prompt

def build_prompt_with_type_hint(row, attribute, candidates=None):
    """Prompt B: includes type hint for numeric attributes."""
    product = {k: v for k, v in row.items()
               if k in ["title", "model", "brand", "product_type"] and pd.notna(v)}
    type_hint = " (numeric value)" if attribute in NUMERIC_ATTRIBUTES else ""
    prompt = (
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"MISSING ATTRIBUTE: {attribute}{type_hint}\n\n"
        f"Respond with VALUE:<answer> only. Example: VALUE:3500"
        if attribute in NUMERIC_ATTRIBUTES else
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"MISSING ATTRIBUTE: {attribute}\n\n"
        f"Respond with VALUE:<answer> only. Example: VALUE:PCIe"
    )
    return prompt

def build_prompt_rag(row, attribute, candidates):
    """Prompt C: RAG prompt with retrieved candidates."""
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
    type_hint = " (numeric value)" if attribute in NUMERIC_ATTRIBUTES else ""
    example = "VALUE:3500" if attribute in NUMERIC_ATTRIBUTES else "VALUE:PCIe"
    return (
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"SIMILAR REFERENCE PRODUCTS:\n{cands_text}\n\n"
        f"MISSING ATTRIBUTE: {attribute}{type_hint}\n\n"
        f"Use the reference products to find the correct value. "
        f"Respond with VALUE:<answer> only. Example: {example}"
    )

PROMPT_BUILDERS = {
    "prompt_basic": build_prompt_basic,
    "prompt_type_hint": build_prompt_with_type_hint,
    "prompt_rag": build_prompt_rag,
}
print("Prompt variants defined:", list(PROMPT_BUILDERS.keys()))


# ## 7. Retriever

# In[25]:


LOCAL_MODEL_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

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

def retrieve(query_row, kb, kb_embeddings, top_k=3):
    q_text = row_to_text(query_row)
    q_emb = embedding_model.encode(q_text, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, kb_embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return kb.iloc[top_idx]

print("Encoding all KBs (clean + noisy)...")
clean_kb_encoded = {}
noisy_kb_encoded = {}
for q in [1, 2, 3, 4]:
    print(f"  Clean KB for query {q}...")
    clean_kb_encoded[q] = encode_kb(clean_kbs[q])
    print(f"  Noisy KB for query {q}...")
    noisy_kb_encoded[q] = encode_kb(noisy_kbs[q])
print("All KBs encoded.")


# ## 8. Response Parser

# In[26]:


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


# ## 9. Evaluation Metrics
# ### 9a. Partial Match (string-based)

# In[27]:


def normalize(val):
    return str(val).lower().strip()

def is_correct_partial(predicted, ground_truth):
    p, g = normalize(predicted), normalize(ground_truth)
    return p == g or p in g or g in p

def is_correct_numeric(predicted, ground_truth, tolerance=0.10):
    """For numeric attributes: correct if within 10% of ground truth."""
    try:
        p = float(str(predicted).replace(',', '').strip())
        g = float(str(ground_truth).replace(',', '').strip())
        if g == 0:
            return p == 0
        return abs(p - g) / abs(g) <= tolerance
    except:
        return is_correct_partial(predicted, ground_truth)

def is_correct(predicted, ground_truth, attribute):
    if predicted == "UNKNOWN":
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        return is_correct_numeric(predicted, ground_truth)
    return is_correct_partial(predicted, ground_truth)


# ### 9b. LLM-Based Evaluation

# In[28]:


def llm_evaluate(predicted, ground_truth, attribute, llm):
    """
    Ask the LLM to judge whether the predicted value is correct,
    acceptable, or wrong compared to the ground truth.
    Returns: 'correct', 'acceptable', or 'wrong'
    """
    if predicted == "UNKNOWN":
        return "wrong"

    prompt = (
        f"You are evaluating a data cleaning prediction.\n\n"
        f"Attribute: {attribute}\n"
        f"Ground truth: {ground_truth}\n"
        f"Predicted: {predicted}\n\n"
        f"Judge the prediction as one of:\n"
        f"- CORRECT: exact match or equivalent meaning\n"
        f"- ACCEPTABLE: more detailed or slightly different but not wrong "
        f"(e.g. 'PCIe 3.0 x16' when GT is 'PCIe 3.0')\n"
        f"- WRONG: incorrect value\n\n"
        f"Respond with JUDGMENT:<label> only. Example: JUDGMENT:CORRECT"
    )
    response = llm.generate(prompt)
    for line in response.splitlines():
        line = line.strip().upper()
        if line.startswith("JUDGMENT:"):
            label = line.split(":", 1)[1].strip()
            if label in {"CORRECT", "ACCEPTABLE", "WRONG"}:
                return label.lower()
    # Fallback: check raw response
    r = response.upper()
    if "CORRECT" in r:
        return "correct"
    if "ACCEPTABLE" in r:
        return "acceptable"
    return "wrong"


# ## 10. Evaluation Runner

# In[29]:


def run_evaluation(eval_df, dfs, kb_encoded_dict, config_name,
                   prompt_builder, use_rag, top_k=3,
                   use_llm_eval=False, llm=None,
                   save_path=None):
    results = []
    total = len(eval_df)
    config_start = time.time()

    # Precompute query embeddings per query dataset
    query_embeddings = {}
    if use_rag:
        for q in eval_df["query_dataset"].unique():
            subset = eval_df[eval_df["query_dataset"] == q]
            rows = [dfs[q].loc[task["df_idx"]] for _, task in subset.iterrows()]
            texts = [row_to_text(r) for r in rows]
            print(f"Precomputing query embeddings for dataset {q}...")
            query_embeddings[q] = embedding_model.encode(
                texts, convert_to_tensor=True,
                batch_size=64, show_progress_bar=True
            )

    # Index within each query dataset for embedding lookup
    query_counters = {q: 0 for q in [1,2,3,4]}

    for i, (_, task) in enumerate(eval_df.iterrows()):
        t0 = time.time()
        q = task["query_dataset"]
        row = dfs[q].loc[task["df_idx"]]
        attr = task["attribute"]
        gt = task["ground_truth"]

        if use_rag:
            kb, kb_emb = kb_encoded_dict[q]
            emb_idx = query_counters[q]
            q_emb = query_embeddings[q][emb_idx]
            query_counters[q] += 1
            scores = util.cos_sim(q_emb, kb_emb)[0]
            top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
            candidates = kb.iloc[top_idx]
            prompt = prompt_builder(row, attr, candidates)
        else:
            prompt = prompt_builder(row, attr)
            query_counters[q] += 1

        predicted = parse_response(llm.generate(prompt))

        # String-based scoring
        correct_partial = is_correct(predicted, gt, attr)

        # LLM-based scoring (optional, slower)
        llm_judgment = None
        if use_llm_eval and llm is not None:
            llm_judgment = llm_evaluate(predicted, gt, attr, llm)

        results.append({
            "config": config_name,
            "query_dataset": q,
            "df_idx": task["df_idx"],
            "attribute": attr,
            "is_numeric": task["is_numeric"],
            "ground_truth": gt,
            "predicted": predicted,
            "correct_partial": correct_partial,
            "llm_judgment": llm_judgment,
            "unknown": predicted == "UNKNOWN"
        })

        if i % 20 == 0:
            elapsed = time.time() - config_start
            eta = (elapsed / (i+1)) * (total - i - 1)
            print(f"[{config_name}] {i+1}/{total} | "
                  f"{time.time()-t0:.2f}s | "
                  f"ETA: {eta/60:.1f} min | "
                  f"D{q} {attr}")

    results_df = pd.DataFrame(results)
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"✓ Saved [{config_name}] → {save_path}")
    return results_df


# ## 11. Run All Configurations
# ### 11a. LLM-only (no retrieval)

# In[32]:


results_llm = run_evaluation(
    eval_df, dfs, clean_kb_encoded,
    config_name="LLM-only",
    prompt_builder=build_prompt_with_type_hint,
    use_rag=False, llm=llm,
    save_path="results_exp4_llm_only.csv"
)


# ### 11b. RAG — Clean KB, top_k=3

# In[ ]:


results_rag_clean = run_evaluation(
    eval_df, dfs, clean_kb_encoded,
    config_name="RAG-clean",
    prompt_builder=build_prompt_rag,
    use_rag=True, top_k=3, llm=llm,
    save_path="results_exp4_rag_clean.csv"
)


# ### 11c. RAG — Noisy KB, top_k=3

# In[ ]:


results_rag_noisy = run_evaluation(
    eval_df, dfs, noisy_kb_encoded,
    config_name="RAG-noisy",
    prompt_builder=build_prompt_rag,
    use_rag=True, top_k=3, llm=llm,
    save_path="results_exp4_rag_noisy.csv"
)


# ### 11d. RAG — Clean KB, top_k=5

# In[ ]:


results_rag_clean_k5 = run_evaluation(
    eval_df, dfs, clean_kb_encoded,
    config_name="RAG-clean-k5",
    prompt_builder=build_prompt_rag,
    use_rag=True, top_k=5, llm=llm,
    save_path="results_exp4_rag_clean_k5.csv"
)


# ## 12. LLM-Based Evaluation (on a sample)
# Run LLM eval on a 100-row sample to keep it fast

# In[ ]:


# Sample 100 tasks stratified by attribute for LLM eval
sample_idx = (
    eval_df.groupby("attribute")
    .apply(lambda x: x.sample(min(20, len(x)), random_state=42))
    .reset_index(drop=True)
)
print(f"LLM eval sample: {len(sample_idx)} tasks")
print(sample_idx["attribute"].value_counts())

results_llm_eval = run_evaluation(
    sample_idx, dfs, clean_kb_encoded,
    config_name="RAG-clean-LLMeval",
    prompt_builder=build_prompt_rag,
    use_rag=True, top_k=3, llm=llm,
    use_llm_eval=True,
    save_path="results_exp4_llm_eval_sample.csv"
)

# Also run LLM-only on same sample for comparison
results_llm_only_eval = run_evaluation(
    sample_idx, dfs, clean_kb_encoded,
    config_name="LLM-only-LLMeval",
    prompt_builder=build_prompt_with_type_hint,
    use_rag=False, llm=llm,
    use_llm_eval=True,
    save_path="results_exp4_llm_only_eval_sample.csv"
)


# ## 13. Results & Visualizations

# In[ ]:


all_results = pd.concat([
    results_llm, results_rag_clean,
    results_rag_noisy, results_rag_clean_k5
], ignore_index=True)
all_results.to_csv("results_exp4_all.csv", index=False)

print("=== OVERALL ACCURACY (partial match) ===")
print(all_results.groupby("config")["correct_partial"].mean().round(3))

print("\n=== PER-ATTRIBUTE ACCURACY ===")
print(all_results.groupby(["config", "attribute"])["correct_partial"].mean().round(3).unstack())

print("\n=== NUMERIC vs TEXT ACCURACY ===")
print(all_results.groupby(["config", "is_numeric"])["correct_partial"].mean().round(3).unstack())


# In[ ]:


# Overall accuracy bar chart
order = ["LLM-only", "RAG-clean", "RAG-clean-k5", "RAG-noisy"]
overall = all_results.groupby("config")["correct_partial"].mean().reindex(order)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(overall.index, overall.values,
              color=sns.color_palette("muted", len(overall)),
              edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_title("Overall Accuracy by Configuration (Experiment 4)", fontsize=13)
ax.set_xlabel("Configuration", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
plt.tight_layout()
plt.savefig("fig_exp4_overall.png", dpi=150)
plt.show()


# In[ ]:


# Per-attribute grouped bar chart
per_attr = all_results.groupby(["config", "attribute"])["correct_partial"].mean().reset_index()
per_attr["config"] = pd.Categorical(per_attr["config"], categories=order, ordered=True)
pivot = per_attr.pivot(index="attribute", columns="config", values="correct_partial")[order]

fig, ax = plt.subplots(figsize=(13, 6))
pivot.plot(kind="bar", ax=ax, edgecolor="white", width=0.75)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Attribute", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Per-Attribute Accuracy — Clean vs Noisy KB", fontsize=13)
ax.legend(title="Config", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("fig_exp4_per_attribute.png", dpi=150)
plt.show()


# In[ ]:


# LLM evaluation results — correct vs acceptable vs wrong
eval_results = pd.concat([results_llm_eval, results_llm_only_eval], ignore_index=True)
eval_results = eval_results[eval_results["llm_judgment"].notna()]

judgment_counts = (
    eval_results.groupby(["config", "llm_judgment"])
    .size().unstack(fill_value=0)
)
print("\n=== LLM EVALUATION RESULTS ===")
print(judgment_counts)

judgment_pct = judgment_counts.div(judgment_counts.sum(axis=1), axis=0)
col_order = [c for c in ["correct", "acceptable", "wrong"] if c in judgment_pct.columns]
judgment_pct[col_order].plot(
    kind="bar", stacked=True, figsize=(8, 5),
    color=["#4CAF50", "#FF9800", "#F44336"],
    edgecolor="white"
)
plt.title("LLM-Based Evaluation: Judgment Distribution", fontsize=13)
plt.ylabel("Proportion", fontsize=12)
plt.xlabel("Configuration", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Judgment", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig_exp4_llm_eval.png", dpi=150)
plt.show()


# In[ ]:


# Accuracy per query dataset — does performance vary by source?
per_dataset = all_results.groupby(["config", "query_dataset"])["correct_partial"].mean().reset_index()
per_dataset["config"] = pd.Categorical(per_dataset["config"], categories=order, ordered=True)
pivot_ds = per_dataset.pivot(index="query_dataset", columns="config", values="correct_partial")[order]

fig, ax = plt.subplots(figsize=(9, 5))
pivot_ds.plot(kind="bar", ax=ax, edgecolor="white", width=0.7)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Query Dataset", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Accuracy by Query Dataset", fontsize=13)
ax.legend(title="Config", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("fig_exp4_per_dataset.png", dpi=150)
plt.show()

