"""
Experiment 6 — PyDI LLMExtractor (LLM-only, no KB)
====================================================
- 25 query rows from Dataset 1 with missing values
- 5 target attributes: bus_type, model_number, model, read_speed_mb_s, height_mm
- Uses PyDI's LLMExtractor properly
- Partial match scoring against ground truth via cluster_id
"""

import sys
import os
sys.path.append("/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/PyDI")
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import random
from langchain_ollama import ChatOllama
from PyDI.informationextraction.llm import LLMExtractor
from pydantic import BaseModel, Field
from typing import Optional

random.seed(42)
np.random.seed(42)

TARGET_ATTRIBUTES = [
    "bus_type",
    "model_number",
    "model",
    "read_speed_mb_s",
    "height_mm"
]
NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "height_mm"}

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
df2 = pd.read_json("normalized_products/dataset_2_normalized.json")
df3 = pd.read_json("normalized_products/dataset_3_normalized.json")
df4 = pd.read_json("normalized_products/dataset_4_normalized.json")
kb_full = pd.concat([df2, df3, df4], ignore_index=True)
kb = kb_full.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Dataset 1: {len(df1)} rows | KB: {len(kb)} rows (sampled from {len(kb_full)})")

# ── Ground truth via cluster_id ────────────────────────────────────────────────
def get_ground_truth(cluster_id, attribute, kb):
    matches = kb_full[kb_full["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

# ── Find 25 rows that have AT LEAST ONE missing recoverable attribute ──────────
print("Finding 25 query rows...")
selected_rows = []
for idx, row in df1.iterrows():
    missing_attrs = []
    for attr in TARGET_ATTRIBUTES:
        if pd.isna(row.get(attr)):
            gt = get_ground_truth(row["cluster_id"], attr, kb_full)
            if gt is not None:
                missing_attrs.append((attr, gt))
    if missing_attrs:
        selected_rows.append({
            "df1_idx": idx,
            "missing_attrs": missing_attrs
        })
    if len(selected_rows) == 25:
        break

print(f"Found {len(selected_rows)} query rows")

# Build flat eval set
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
print(f"Total eval tasks: {len(eval_df)}")
print(eval_df["attribute"].value_counts())

# ── Get the 25 query rows as a DataFrame ──────────────────────────────────────
query_indices = [item["df1_idx"] for item in selected_rows]
query_df = df1.loc[query_indices].copy()
print(f"\nQuery DataFrame: {len(query_df)} rows")

# ── PyDI LLMExtractor setup ───────────────────────────────────────────────────
print("\nSetting up PyDI LLMExtractor...")
chat_model = ChatOllama(model="llama3.1:8b", temperature=0)

# Test
test = chat_model.invoke("Say OK")
print("LangChain Ollama OK:", repr(test.content[:20]))

class ProductAttributes(BaseModel):
    bus_type: Optional[str] = Field(
        None, description="Interface bus type e.g. PCIe 3.0, SATA III, USB 3.0")
    model_number: Optional[str] = Field(
        None, description="Exact product model number or SKU")
    model: Optional[str] = Field(
        None, description="Product model name")
    read_speed_mb_s: Optional[float] = Field(
        None, description="Sequential read speed in MB/s as a number only")
    height_mm: Optional[float] = Field(
        None, description="Product height in millimeters as a number only")

system_prompt = """You are a product data expert specializing in computer hardware.
Extract the following attributes from the product information.
For numeric attributes (read_speed_mb_s, height_mm), return only the number.
If an attribute cannot be determined from the text, return null.
Return valid JSON only, no explanation."""

extractor = LLMExtractor(
    chat_model=chat_model,
    source_column="title_description",
    system_prompt=system_prompt,
    schema=ProductAttributes,
)

# ── Run extraction ─────────────────────────────────────────────────────────────
print(f"\nRunning PyDI LLMExtractor on {len(query_df)} rows...")
import time
t0 = time.time()
extracted_df = extractor.extract(query_df)
print(f"Extraction completed in {time.time()-t0:.1f}s")
print("\nSample extracted values:")
print(extracted_df[TARGET_ATTRIBUTES].head(5).to_string())

# ── Scoring ────────────────────────────────────────────────────────────────────
def normalize(val):
    return str(val).lower().strip()

def is_correct(predicted, ground_truth, attribute):
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

results = []
for _, task in eval_df.iterrows():
    idx = task["df1_idx"]
    attr = task["attribute"]
    gt = task["ground_truth"]

    if attr in extracted_df.columns and idx in extracted_df.index:
        predicted = extracted_df.loc[idx, attr]
        predicted_str = str(predicted) if pd.notna(predicted) and predicted is not None else "UNKNOWN"
    else:
        predicted_str = "UNKNOWN"

    correct = is_correct(predicted_str, gt, attr)
    unknown = predicted_str == "UNKNOWN"

    print(f"  {attr:<20} | GT: {str(gt):<25} | Pred: {predicted_str:<25} | {'✓' if correct else '✗'}")

    results.append({
        "config": "PyDI-LLMExtractor",
        "df1_idx": idx,
        "attribute": attr,
        "is_numeric": task["is_numeric"],
        "ground_truth": gt,
        "predicted": predicted_str,
        "correct": correct,
        "unknown": unknown
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results_exp6_pydi_llm.csv", index=False)
print("\n✓ Saved results to results_exp6_pydi_llm.csv")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY — PyDI LLMExtractor")
print("="*60)
print(f"Overall accuracy: {results_df['correct'].mean():.3f}")
print(f"UNKNOWN rate:     {results_df['unknown'].mean():.3f}")
print(f"Total tasks:      {len(results_df)}")

print("\nPer-attribute accuracy:")
summary = results_df.groupby("attribute").agg(
    total=("correct", "count"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
    unknown_rate=("unknown", "mean")
).round(3)
print(summary.to_string())
