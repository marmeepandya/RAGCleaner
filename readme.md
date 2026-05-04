# RAG-Driven Data Cleaning with PyDI

A seminar project for **CS 715 Solving Complex Tasks with Large Language Models**  
University of Mannheim, FSS 2026  
**Author:** Marmee Pandya  

---

## Overview

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** approach to automated data cleaning, integrated into the [PyDI framework](https://github.com/wbsg-uni-mannheim/PyDI). Given a product dataset with missing attribute values, the system retrieves semantically similar products from a knowledge base and uses a local LLM to predict the missing values.

The core idea: instead of relying on an LLM's pre-training knowledge alone — which fails for product-specific attributes like model numbers or storage speeds — we ground every prediction in retrieved evidence from a knowledge base of similar products. This follows the **Scenario 3** setup of RetClean (Naeem et al., VLDB 2024): local LLM + retrieval, privacy-preserving, no data leaves the cluster.

---

## Project Structure

```
RAG_Data_Cleaning/
├── normalized_products/           # Input datasets (4 JSON files)
│   ├── dataset_1_normalized.json  # Query set — 812 rows
│   ├── dataset_2_normalized.json  # KB — ~733 rows
│   ├── dataset_3_normalized.json  # KB — ~733 rows
│   └── dataset_4_normalized.json  # KB — ~734 rows
│
├── embeddings/                    # Pre-computed embedding tensors (.pt files)
│   ├── minilm_kb.pt               # all-MiniLM-L6-v2 KB embeddings (384-dim)
│   ├── minilm_query.pt
│   ├── bge_kb.pt                  # BGE-large-en-v1.5 KB embeddings (1024-dim)
│   ├── bge_query.pt
│   ├── openai_kb.pt               # text-embedding-3-large KB embeddings (3072-dim)
│   └── openai_query.pt
│
├── results/                       # Experiment output CSVs
│   ├── exp1_llm_only.csv
│   ├── exp2_rag_minilm.csv
│   ├── exp3_rag_minilm_reranker.csv
│   ├── exp4_rag_bge_reranker.csv
│   ├── exp5_rag_te_reranker.csv
│   ├── master_predictions.csv     # All predictions side by side
│   ├── null_analysis.csv          # Attribute completeness analysis
│   └── error_analysis.csv         # Task-level failure analysis
│
├── figures/                       # Generated figures for report and presentation
│
├── exp_setup.ipynb                # ENTRY POINT — builds eval set + Exp 1 + all embeddings
├── exp_runner_TE.py               # Exp 5 — TE-large + CrossEncoder reranker
├── error_analysis_plots.ipynb     # Full error analysis + 12 figures
│
├── eval_set.csv                   # Canonical 96-task eval set (generated once)
├── query_indices.csv              # Query row indices for eval set
│
└── README.md                      # This file
```

---

## Dataset

Four product offer datasets (GPUs, SSDs, HDDs, USB drives) from the PyDI project. Products across datasets are linked by a `cluster_id` identifying matching offers for the same real-world product.

| Dataset | Role | Rows |
|---|---|---|
| Dataset 1 | Query set (to clean) | 812 |
| Dataset 2 | Knowledge base | ~733 |
| Dataset 3 | Knowledge base | ~733 |
| Dataset 4 | Knowledge base | ~734 |
| **Combined KB** | **Knowledge base** | **2,200** |

**Important:** All 96 evaluation tasks were selected only where the ground truth value is confirmed present in the KB (100% coverage). The model is never penalised for values that do not exist anywhere in the KB.

---

## Target Attributes

7 attributes selected to cover a range of difficulty levels:

| Attribute | Type | Tasks | LLM Difficulty | Why |
|---|---|---|---|---|
| `bus_type` | text | 13 | Easy | LLM knows PCIe / SATA / USB |
| `model` | text | 10 | Medium | Partially in descriptions |
| `model_number` | text | 23 | Hard | Exact SKU — cannot guess |
| `read_speed_mb_s` | numeric | 15 | **Impossible** | Never in product descriptions |
| `write_speed_mb_s` | numeric | 10 | **Impossible** | Never in product descriptions |
| `height_mm` | numeric | 13 | **Impossible** | Must retrieve from KB |
| `width_mm` | numeric | 12 | **Impossible** | Must retrieve from KB |
| **Total** | | **96** | | |

---

## Approach: 5 Configurations

Each configuration adds exactly one component. Same 96-task eval set used across all experiments.

| Config | Description |
|---|---|
| **Exp 1 — LLM-only** | Llama 3.1 8B predicts from product title only. No KB access. Baseline. |
| **Exp 2 — RAG-MiniLM** | MiniLM-L6 (384-dim) encodes KB + query. Top-3 by cosine similarity passed to Llama. |
| **Exp 3 — MiniLM + Reranker** | MiniLM retrieves top-20. CrossEncoder reranks to top-5. RetClean-inspired two-stage pipeline. |
| **Exp 4 — BGE + Reranker** | Replaces MiniLM with BGE-large-en-v1.5 (1024-dim, top MTEB). Adds description field. |
| **Exp 5 — TE + Reranker** | Uses OpenAI text-embedding-3-large (3072-dim). Strongest embedding model tested. |

---

## System Architecture

**Saved to disk**
```
KB 2,200 rows → BGE-large Encode → Dense Vectors (1024-dim) → Save bge_kb.pt
```

**Per Query**
```
Query Product → BGE-large Encode → Cosine Similarity (loads bge_kb.pt)
→ Top-20 Candidates → CrossEncoder Re-rank → Top-5
→ Few-shot Prompt → Llama 3.1 8B → VALUE:<answer> → Predicted Value
```

**Prompting strategy:** Few-shot match-then-extract. The LLM is instructed to first identify the best matching reference product, then copy the attribute value exactly. Strict grounding instructions prevent the LLM from using its own knowledge. Uncertain cases return `VALUE:UNKNOWN`.

---

## Evaluation

- **Standard accuracy** — exact match for numeric attributes (source variation confirmed 0% across all 96 tasks), substring match for text attributes
- **CE eval** — CrossEncoder semantic evaluation for text attributes; exact match for numeric with ±10% tolerance classified as acceptable (removes self-evaluation bias of LLM-as-judge)
- **UNKNOWN rate** — fraction of tasks where the model declined to predict
- **Retrieval metrics** — Recall@K, Precision@K, NDCG@K computed across all embedding models

> **Note on numeric evaluation:** We initially used ±10% tolerance for numeric attributes. After verifying empirically that all KB values for the same cluster are identical across all datasets (0% source variation, 50/50 consistent), we switched to exact match as the primary standard accuracy metric. The ±10% window is retained only as the "acceptable" threshold in CE eval.

---

## Results

| Config | Standard Acc | CE Eval | UNKNOWN Rate |
|---|---|---|---|
| LLM-only | 17.7% | 21.9% | 44.8% |
| RAG-MiniLM | 41.7% | 44.8% | 17.7% |
| MiniLM+RR | 74.0% | 76.0% | 8.3% |
| BGE+RR | 76.0% | 78.1% | 3.1% |
| **TE+RR** | **77.1%** | **83.3%** | **2.1%** |

Key findings:
- **4.4× improvement** from LLM-only to best RAG config
- Numeric attributes: **0% → 70–90%** — impossible without KB, straightforward with retrieval
- CrossEncoder reranking specifically fixes `model_number` (18% → 91% CE eval)
- BGE+RR achieves lowest UNKNOWN rate (3.1%) among retrieval configs
- Remaining failures: near-identical SKU confusion (~35%), cluster fragmentation (~20%), field confusion (~20%)

---

## Setup

### Requirements
```bash
pip install sentence-transformers pandas numpy torch matplotlib seaborn scikit-learn langchain-ollama openai --break-system-packages
```

### Ollama (LLM inference)
```bash
ollama pull llama3.1:8b
```

### On bwUniCluster 3.0
```bash
# Load modules
module load cs/ollama/0.5.11
module load devel/cuda/12.8

# Start Ollama on non-default port (avoids conflicts)
OLLAMA_HOST=127.0.0.1:11435 ollama serve &
sleep 15

# Activate venv
source /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/bin/activate
```

### SLURM batch job
```bash
sbatch run_experiment.sh
```

Partition: `gpu_a100_short` (30 min limit). All experiment scripts use checkpoint system — saves every 5 predictions, resumes automatically on timeout.

### Running experiments

**Step 1 — Always run first (builds eval set + embeddings):**
```bash
jupyter nbconvert --to notebook --execute exp_setup.ipynb
```

**Step 2 — Run individual experiments:**
```bash
python exp_runner_TE.py          # Exp 5: TE-large + CrossEncoder
```

**Step 3 — Error analysis and figures:**
```bash
jupyter nbconvert --to notebook --execute error_analysis_plots.ipynb
```

---

## Reproducibility

The following measures are taken to maximise reproducibility:
- `temperature=0` and `seed=42` for Ollama inference
- Eval set saved to `eval_set.csv` once — all experiments reload the same 96 tasks
- Embeddings saved as `.pt` files — same float tensors reused across all runs
- Checkpoint system saves every 5 predictions — safe to resume after GPU timeout
- Remaining non-determinism: GPU floating point rounding on A100 (unavoidable, ~5% variance across separate job submissions)

---

## References

- Naeem et al. (2024). *RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes*. PVLDB 17(12).
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Xiao et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding*. (BGE-large)
- Narayan et al. (2022). *Can Foundation Models Wrangle Your Data?* PVLDB 16(4).