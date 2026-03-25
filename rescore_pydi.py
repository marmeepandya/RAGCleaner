import sys
sys.path.append("/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/PyDI")
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from PyDI.informationextraction.llm import LLMExtractor
from pydantic import BaseModel, Field
from typing import Optional

TARGET_ATTRIBUTES = [
    "bus_type", "model_number", "model",
    "read_speed_mb_s", "write_speed_mb_s",
    "height_mm", "width_mm"
]
NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}

df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
eval_df_ref = pd.read_csv("results_exp5_llm_only.csv")

# Get the 25 sampled row indices
sampled_rows = eval_df_ref["df1_idx"].unique() if "df1_idx" in eval_df_ref.columns else None

# Rebuild eval_df with df1_idx
import random, numpy as np
random.seed(42); np.random.seed(42)

def get_ground_truth(cluster_id, attribute, kb):
    matches = kb[kb["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

df2 = pd.read_json("normalized_products/dataset_2_normalized.json")
df3 = pd.read_json("normalized_products/dataset_3_normalized.json")
df4 = pd.read_json("normalized_products/dataset_4_normalized.json")
kb = pd.concat([df2, df3, df4], ignore_index=True)

eval_records = []
for idx, row in df1.iterrows():
    for attr in TARGET_ATTRIBUTES:
        if pd.isna(row.get(attr)):
            gt = get_ground_truth(row["cluster_id"], attr, kb)
            if gt is not None:
                eval_records.append({
                    "df1_idx": idx,
                    "attribute": attr,
                    "ground_truth": gt,
                    "is_numeric": attr in NUMERIC_ATTRIBUTES
                })

eval_df = pd.DataFrame(eval_records)
unique_rows = eval_df["df1_idx"].unique()
sampled_rows = pd.Series(unique_rows).sample(n=min(25, len(unique_rows)), random_state=42).values
eval_df = eval_df[eval_df["df1_idx"].isin(sampled_rows)].reset_index(drop=True)

print(f"Eval set: {len(eval_df)} tasks")

# Run LLMExtractor once per row, score per attribute
chat_model = ChatOllama(model="llama3.1:8b", temperature=0)

class ProductAttributes(BaseModel):
    bus_type: Optional[str] = Field(None)
    model_number: Optional[str] = Field(None)
    model: Optional[str] = Field(None)
    read_speed_mb_s: Optional[float] = Field(None)
    write_speed_mb_s: Optional[float] = Field(None)
    height_mm: Optional[float] = Field(None)
    width_mm: Optional[float] = Field(None)

extractor = LLMExtractor(
    chat_model=chat_model,
    source_column="title_description",
    system_prompt=(
        "You are a product data expert. Extract ALL of the following attributes "
        "from the product text. Return null for any attribute not found. "
        "Return valid JSON only."
    ),
    schema=ProductAttributes,
)

sampled_df1 = df1.loc[sampled_rows].copy()
print(f"Running LLMExtractor on {len(sampled_df1)} rows...")
extracted_df = extractor.extract(sampled_df1)
print("Extraction done.")
print(extracted_df[TARGET_ATTRIBUTES].head())

def is_correct(predicted, ground_truth, attribute):
    if predicted is None or str(predicted).strip() in {"", "nan", "None", "UNKNOWN"}:
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

results = []
for _, task in eval_df.iterrows():
    idx = task["df1_idx"]
    attr = task["attribute"]
    gt = task["ground_truth"]
    if attr in extracted_df.columns and idx in extracted_df.index:
        predicted = extracted_df.loc[idx, attr]
        predicted = str(predicted) if pd.notna(predicted) and predicted is not None else "UNKNOWN"
    else:
        predicted = "UNKNOWN"
    correct = is_correct(predicted, gt, attr)
    results.append({
        "config": "PyDI-LLMExtractor",
        "df1_idx": idx,
        "attribute": attr,
        "is_numeric": task["is_numeric"],
        "ground_truth": gt,
        "predicted": predicted,
        "correct": correct,
        "unknown": predicted == "UNKNOWN"
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results_exp5_pydi_llmextractor.csv", index=False)

print("\n=== PyDI LLMExtractor Results ===")
print(f"Overall accuracy: {results_df['correct'].mean():.3f}")
print(f"UNKNOWN rate: {results_df['unknown'].mean():.3f}")
print("\nPer attribute:")
print(results_df.groupby("attribute")["correct"].mean().round(3))
