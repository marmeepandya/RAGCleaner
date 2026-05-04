import re
import pandas as pd
import numpy as np
import glob
from sentence_transformers import CrossEncoder

# ── Load CrossEncoder for CE eval ─────────────────────────────────────────────
CE_CACHE = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/"
ce_snaps = glob.glob(CE_CACHE + "*/")
CE_PATH  = ce_snaps[0].rstrip("/") if ce_snaps else "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(CE_PATH)

NUMERIC_ATTRIBUTES = {"read_speed_mb_s", "write_speed_mb_s", "height_mm", "width_mm"}

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
            if g == 0: return "correct" if p == 0 else "wrong"
            ratio = abs(p - g) / abs(g)
            if ratio <= 0.10: return "correct"
            if ratio <= 0.30: return "acceptable"
            return "wrong"
        except:
            return "wrong"
    score = cross_encoder.predict([[ground_truth, predicted]])[0]
    if score > 2.0:  return "correct"
    if score > -1.0: return "acceptable"
    return "wrong"

# ── Rest of your code unchanged ───────────────────────────────────────────────
results_df = pd.read_csv("exp10_bge_rag.csv")

def fix_prediction(pred):
    if isinstance(pred, str) and pred.strip().upper().startswith("VALUE:"):
        val = pred.strip().split(":", 1)[1].strip()
        if val.upper() in {"UNKNOWN", "NONE", "NULL", "NAN", ""}:
            return "UNKNOWN"
        return val
    return pred

results_df["predicted"]       = results_df["predicted"].apply(fix_prediction)
results_df["unknown"]         = results_df["predicted"] == "UNKNOWN"
results_df["correct_standard"] = results_df.apply(
    lambda row: is_correct_standard(row["predicted"], row["ground_truth"], row["attribute"]), axis=1
)

print("Re-evaluating...")
results_df["ce_judgment"] = [
    evaluate_prediction(row["predicted"], row["ground_truth"], row["attribute"])
    for _, row in results_df.iterrows()
]

results_df.to_csv("exp10_bge_rag.csv", index=False)

std = results_df["correct_standard"].mean()
ce  = (results_df["ce_judgment"].isin(["correct","acceptable"])).mean()
unk = results_df["unknown"].mean()
print(f"Standard: {std:.3f}  CE: {ce:.3f}  UNKNOWN: {unk:.3f}")
print(results_df.groupby("attribute").agg(std_acc=("correct_standard","mean"), unknown=("unknown","mean")).round(3).to_string())