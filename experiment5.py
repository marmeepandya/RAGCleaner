"""
Experiment 5 — PyDI LLMExtractor vs RAGCleaner
===============================================
Compares three approaches:
1. PyDI LLMExtractor  — extract from description using LangChain + Ollama (no KB)
2. RAGCleaner         — retrieve from KB + LLM (your extension of PyDI)
3. LLM-only baseline  — direct Ollama call, no KB, no PyDI

All three are evaluated on the same 25-row sample from Dataset 1.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.append("/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/PyDI")

import pandas as pd
import numpy as np
import requests
import time
import random
from sentence_transformers import SentenceTransformer, util

# PyDI imports
from cleaners.rag_cleaner import RAGCleaner
from PyDI.informationextraction.llm import LLMExtractor

# LangChain + Ollama
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional

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
kb = pd.concat([df2, df3, df4], ignore_index=True)
print(f"Dataset 1: {len(df1)} rows | KB: {len(kb)} rows")

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
            gt = get_ground_truth(row["cluster_id"], attr, kb)
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

# Sample 25 unique rows
unique_rows = eval_df["df1_idx"].unique()
sampled_rows = pd.Series(unique_rows).sample(
    n=min(25, len(unique_rows)), random_state=42).values
eval_df = eval_df[eval_df["df1_idx"].isin(
    sampled_rows)].reset_index(drop=True)
print(f"Sampled eval set: {len(eval_df)} tasks across 25 rows")
print(eval_df["attribute"].value_counts())

# ── Scoring ────────────────────────────────────────────────────────────────────
def normalize(val):
    return str(val).lower().strip()

def is_correct(predicted, ground_truth, attribute):
    if predicted is None or str(predicted).strip() == "" or predicted == "UNKNOWN":
        return False
    if attribute in NUMERIC_ATTRIBUTES:
        try:
            p = float(str(predicted).replace(",", "").strip())
            g = float(str(ground_truth).replace(",", "").strip())
            return abs(p - g) / abs(g) <= 0.10 if g != 0 else p == 0
        except:
            pass
    p, g = normalize(str(predicted)), normalize(str(ground_truth))
    return p == g or p in g or g in p

# ── LLM setup ─────────────────────────────────────────────────────────────────
print("\nSetting up LangChain Ollama...")
chat_model = ChatOllama(model="llama3.1:8b", temperature=0)

# Test connection
test = chat_model.invoke("Say OK")
print("LangChain Ollama OK:", repr(test.content[:30]))

# ── CONFIG 1: PyDI LLMExtractor ───────────────────────────────────────────────
print("\n" + "="*60)
print("CONFIG 1: PyDI LLMExtractor (no KB)")
print("="*60)

# Pydantic schema for all target attributes
class ProductAttributes(BaseModel):
    bus_type: Optional[str] = Field(None, description="Bus type e.g. PCIe, SATA, USB")
    model_number: Optional[str] = Field(None, description="Model number/SKU")
    model: Optional[str] = Field(None, description="Product model name")
    read_speed_mb_s: Optional[float] = Field(None, description="Read speed in MB/s")
    write_speed_mb_s: Optional[float] = Field(None, description="Write speed in MB/s")
    height_mm: Optional[float] = Field(None, description="Height in mm")
    width_mm: Optional[float] = Field(None, description="Width in mm")

system_prompt = """You are a product data expert. Extract the following attributes 
from the product information provided. If an attribute cannot be determined from 
the text, set it to null. Return valid JSON matching the schema."""

extractor = LLMExtractor(
    chat_model=chat_model,
    source_column="title_description",
    system_prompt=system_prompt,
    schema=ProductAttributes,
)

# Run extraction on the 25 sampled rows
sampled_df1 = df1.loc[sampled_rows].copy()
print(f"Running PyDI LLMExtractor on {len(sampled_df1)} rows...")
t0 = time.time()
extracted_df = extractor.extract(sampled_df1)
print(f"Extraction done in {time.time()-t0:.1f}s")

# Score PyDI results
pydi_results = []
for _, task in eval_df.iterrows():
    idx = task["df1_idx"]
    attr = task["attribute"]
    gt = task["ground_truth"]
    predicted = extracted_df.loc[idx, attr] if attr in extracted_df.columns else None
    predicted = str(predicted) if pd.notna(predicted) and predicted is not None else "UNKNOWN"
    correct = is_correct(predicted, gt, attr)
    pydi_results.append({
        "config": "PyDI-LLMExtractor",
        "attribute": attr,
        "is_numeric": task["is_numeric"],
        "ground_truth": gt,
        "predicted": predicted,
        "correct": correct,
        "unknown": predicted == "UNKNOWN"
    })

results_pydi = pd.DataFrame(pydi_results)
results_pydi.to_csv("results_exp5_pydi_llmextractor.csv", index=False)
print(f"✓ Saved PyDI-LLMExtractor results")
print(f"Accuracy: {results_pydi['correct'].mean():.3f}")

# ── CONFIG 2: LLM-only baseline (direct Ollama) ───────────────────────────────
print("\n" + "="*60)
print("CONFIG 2: LLM-only baseline (direct Ollama)")
print("="*60)

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

llm_only_results = []
rows = [df1.loc[task["df1_idx"]] for _, task in eval_df.iterrows()]
total = len(eval_df)
t0 = time.time()

for i, (_, task) in enumerate(eval_df.iterrows()):
    row = rows[i]
    attr = task["attribute"]
    gt = task["ground_truth"]
    example = "VALUE:3500" if attr in NUMERIC_ATTRIBUTES else "VALUE:PCIe"
    type_hint = " (numeric)" if attr in NUMERIC_ATTRIBUTES else ""
    product = {k: v for k, v in row.items()
               if k in ["title", "model", "brand", "product_type"] and pd.notna(v)}
    prompt = (
        f"You are a product data expert. Fill in the missing attribute.\n\n"
        f"PRODUCT: {product}\n\n"
        f"MISSING ATTRIBUTE: {attr}{type_hint}\n\n"
        f"Respond with VALUE:<answer> only. Example: {example}"
    )
    predicted = parse_response(llm.generate(prompt))
    correct = is_correct(predicted, gt, attr)
    elapsed = time.time() - t0
    remaining = (elapsed / (i+1)) * (total - i - 1)
    print(f"[LLM-only] {i+1}/{total} | ETA: {remaining/60:.1f}min | "
          f"{attr} | {predicted} | {'✓' if correct else '✗'}")
    llm_only_results.append({
        "config": "LLM-only",
        "attribute": attr,
        "is_numeric": task["is_numeric"],
        "ground_truth": gt,
        "predicted": predicted,
        "correct": correct,
        "unknown": predicted == "UNKNOWN"
    })

results_llm = pd.DataFrame(llm_only_results)
results_llm.to_csv("results_exp5_llm_only.csv", index=False)
print(f"✓ Saved LLM-only results")
print(f"Accuracy: {results_llm['correct'].mean():.3f}")

# ── CONFIG 3: RAGCleaner (PyDI extension) ─────────────────────────────────────
print("\n" + "="*60)
print("CONFIG 3: RAGCleaner — PyDI extension with KB retrieval")
print("="*60)

print("Loading embedding model...")
embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)

print("Encoding KB...")
rag_cleaner = RAGCleaner(
    knowledge_base=kb,
    llm=llm,
    top_k=3,
)

def row_to_text(row):
    attrs = ["title", "model", "model_number", "brand", "product_type"]
    return " | ".join([str(row[a]) for a in attrs if pd.notna(row.get(a))])

print("Precomputing query embeddings...")
query_texts = [row_to_text(r) for r in rows]
query_embeddings = embedding_model.encode(
    query_texts, convert_to_tensor=True,
    batch_size=64, show_progress_bar=True
)

rag_results = []
t0 = time.time()

for i, (_, task) in enumerate(eval_df.iterrows()):
    row = rows[i]
    attr = task["attribute"]
    gt = task["ground_truth"]

    # Retrieve top-k candidates
    scores = util.cos_sim(query_embeddings[i], rag_cleaner.kb_embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:rag_cleaner.top_k]
    candidates = rag_cleaner.kb.iloc[top_idx]
    prompt = rag_cleaner._build_prompt(row, candidates, attr)
    predicted = rag_cleaner._parse_response(llm.generate(prompt))
    correct = is_correct(predicted, gt, attr)
    elapsed = time.time() - t0
    remaining = (elapsed / (i+1)) * (total - i - 1)
    print(f"[RAGCleaner] {i+1}/{total} | ETA: {remaining/60:.1f}min | "
          f"{attr} | {predicted} | {'✓' if correct else '✗'}")
    rag_results.append({
        "config": "RAGCleaner",
        "attribute": attr,
        "is_numeric": task["is_numeric"],
        "ground_truth": gt,
        "predicted": predicted,
        "correct": correct,
        "unknown": predicted == "UNKNOWN"
    })

results_rag = pd.DataFrame(rag_results)
results_rag.to_csv("results_exp5_rag_cleaner.csv", index=False)
print(f"✓ Saved RAGCleaner results")
print(f"Accuracy: {results_rag['correct'].mean():.3f}")

# ── Final Summary ──────────────────────────────────────────────────────────────
all_results = pd.concat([results_pydi, results_llm, results_rag], ignore_index=True)
all_results.to_csv("results_exp5_all.csv", index=False)

print("\n" + "="*60)
print("OVERALL ACCURACY")
print("="*60)
print(all_results.groupby("config")["correct"].mean().round(3))

print("\n" + "="*60)
print("PER-ATTRIBUTE ACCURACY")
print("="*60)
print(all_results.groupby(["config", "attribute"])["correct"].mean().round(3).unstack())

print("\n" + "="*60)
print("NUMERIC vs TEXT")
print("="*60)
print(all_results.groupby(["config", "is_numeric"])["correct"].mean().round(3).unstack())

print("\n" + "="*60)
print("UNKNOWN RATE")
print("="*60)
print(all_results.groupby("config")["unknown"].mean().round(3))

print("\nDone! Results saved to results_exp5_all.csv")
