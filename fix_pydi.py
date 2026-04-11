import sys, os
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

TARGET_ATTRIBUTES = ["bus_type", "model_number", "model",
                     "read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"]
NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}

# Load data
df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
df2 = pd.read_json("normalized_products/dataset_2_normalized.json")
df3 = pd.read_json("normalized_products/dataset_3_normalized.json")
df4 = pd.read_json("normalized_products/dataset_4_normalized.json")
kb = pd.concat([df2, df3, df4], ignore_index=True)

def get_ground_truth(cluster_id, attribute, kb):
    matches = kb[kb["cluster_id"] == cluster_id]
    for _, row in matches.iterrows():
        val = row.get(attribute)
        if pd.notna(val) and str(val).strip().lower() not in {"", "none", "nan"}:
            return str(val).strip()
    return None

# Build eval set
eval_records = []
for idx, row in df1.iterrows():
    for attr in TARGET_ATTRIBUTES:
        if pd.isna(row.get(attr)):
            gt = get_ground_truth(row["cluster_id"], attr, kb)
            if gt is not None:
                eval_records.append({"df1_idx": idx, "attribute": attr,
                                     "ground_truth": gt,
                                     "is_numeric": attr in NUMERIC_ATTRIBUTES})

eval_df = pd.DataFrame(eval_records)
unique_rows = eval_df["df1_idx"].unique()
sampled_rows = pd.Series(unique_rows).sample(n=min(25, len(unique_rows)), random_state=42).values
eval_df = eval_df[eval_df["df1_idx"].isin(sampled_rows)].reset_index(drop=True)
print(f"Eval set: {len(eval_df)} tasks, sampled rows: {sorted(sampled_rows)[:5]}...")

# Run LLMExtractor on sampled rows
sampled_df1 = df1.loc[sampled_rows].copy()
print(f"sampled_df1 index: {sorted(sampled_df1.index)[:5]}...")

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

print("Running extraction...")
extracted_df = extractor.extract(sampled_df1)
print(f"extracted_df index: {sorted(extracted_df.index)[:5]}...")
print(extracted_df[["bus_type", "model_number"]].head(3))

# Score
def is_correct(predicted, ground_truth, attribute):
    if predicted is None or str(predicted).strip() in {"", "nan", "None", "UNKNOWN", "none"}:
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
    # Use .loc with original index
    if attr in extracted_df.columns and idx in extracted_df.index:
        predicted = extracted_df.loc[idx, attr]
        predicted = str(predicted) if pd.notna(predicted) and predicted is not None else "UNKNOWN"
    else:
        predicted = "UNKNOWN"
        print(f"  WARNING: idx={idx} not in extracted_df or attr={attr} missing")
    correct = is_correct(predicted, gt, attr)
    results.append({"config": "PyDI-LLMExtractor", "df1_idx": idx,
                    "attribute": attr, "is_numeric": task["is_numeric"],
                    "ground_truth": gt, "predicted": predicted,
                    "correct": correct, "unknown": predicted == "UNKNOWN"})

results_df = pd.DataFrame(results)
results_df.to_csv("results_exp5_pydi_llmextractor.csv", index=False)

print("\n=== PyDI LLMExtractor Results ===")
print(f"Overall accuracy: {results_df['correct'].mean():.3f}")
print(f"UNKNOWN rate: {results_df['unknown'].mean():.3f}")
print(results_df.groupby("attribute")[["correct", "unknown"]].mean().round(3))

# Final comparison with other configs
results_llm = pd.read_csv("results_exp5_llm_only.csv")
results_rag = pd.read_csv("results_exp5_rag_cleaner.csv")
all_results = pd.concat([results_df, results_llm, results_rag], ignore_index=True)
all_results.to_csv("results_exp5_all.csv", index=False)

print("\n=== FINAL COMPARISON ===")
print(all_results.groupby("config")["correct"].mean().round(3))
print("\nPer attribute:")
print(all_results.groupby(["config", "attribute"])["correct"].mean().round(3).unstack())
