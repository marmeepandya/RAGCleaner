import sys
sys.path.append("/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/PyDI")
import pandas as pd
import numpy as np

# Reload results
results_llm = pd.read_csv("results_exp5_llm_only.csv")
results_rag = pd.read_csv("results_exp5_rag_cleaner.csv")

# Load extracted df
df1 = pd.read_json("normalized_products/dataset_1_normalized.json")
eval_df = pd.read_csv("results_exp5_llm_only.csv")[["attribute", "ground_truth"]].copy()

# Reload the PyDI extracted results
extracted_df = pd.read_csv("results_exp5_pydi_llmextractor.csv")
print("PyDI raw predictions:")
print(extracted_df[["attribute", "ground_truth", "predicted"]].head(10))
