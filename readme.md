# RAG-Driven Data Cleaning with PyDI

A seminar project for **CS 715 — Solving Complex Tasks with Large Language Models**  
University of Mannheim, FSS 2026  
**Author:** Marmee Pandya  
**Supervisor:** Dr. Ralph Peeters

---

## Overview

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** approach to data cleaning, integrated into the [PyDI framework](https://github.com/wbsg-uni-mannheim/PyDI). Given a product dataset with missing attribute values, the system retrieves semantically similar products from a knowledge base and uses an LLM to predict the missing values.

The core idea: instead of relying purely on an LLM's pre-training knowledge (which fails for product-specific attributes like model numbers), we ground predictions in retrieved evidence from a knowledge base of similar products.

---

## Project Structure

```
RAG_Data_Cleaning/
├── PyDI/                          # PyDI framework (submodule)
│   └── PyDI/cleaners/
│       └── rag_cleaner.py         # RAG cleaner implementation
├── normalized_products/           # Input datasets
│   ├── dataset_1_normalized.json
│   ├── dataset_2_normalized.json
│   ├── dataset_3_normalized.json
│   └── dataset_4_normalized.json
├── experiment2.ipynb              # Initial 25-row pilot experiment
├── experiment3.ipynb              # Full evaluation with top-k tuning
├── experiment4.ipynb              # Redesigned eval: rotating KB, noisy KB, LLM eval
├── experiment4.py                 # experiment4 converted to script for cluster
├── error_analysis.ipynb           # Error pattern analysis
├── run_experiment4.sh             # SLURM batch job script
├── results_exp3_*.csv             # Experiment 3 results
├── results_exp4_*.csv             # Experiment 4 results
├── fig_*.png                      # Generated figures
├── report.tex                     # Seminar report (LaTeX)
├── references.bib                 # Bibliography
└── README.md                      # This file
```

---

## Dataset

Four product offer datasets (GPUs, SSDs, HDDs, USB drives) provided as part of the PyDI project. Products across datasets are linked by a `cluster_id` attribute identifying matching offers for the same real-world product.

| Dataset | Role | Rows |
|---|---|---|
| Dataset 1 | Query (to clean) | 812 |
| Dataset 2 | Knowledge base | ~730 |
| Dataset 3 | Knowledge base | ~730 |
| Dataset 4 | Knowledge base | ~730 |
| Combined KB | Knowledge base | 2,200 |

---

## Target Attributes

Selected to cover a range of difficulty levels:

| Attribute | Type | Avg. Recovery | Difficulty |
|---|---|---|---|
| `bus_type` | text | 86% | easy — LLM baseline |
| `model` | text | 75% | medium |
| `model_number` | text | 66% | hard — product-specific |
| `read_speed_mb_s` | numeric | 20% | very hard |
| `write_speed_mb_s` | numeric | 14% | very hard |
| `height_mm` | numeric | 14% | very hard |
| `width_mm` | numeric | 12% | very hard |

---

## Approach

### RAG Pipeline
1. **Retriever** — encode query product using `sentence-transformers/all-MiniLM-L6-v2`, retrieve top-k most similar products from KB by cosine similarity
2. **Prompt builder** — format retrieved candidates + incomplete product into a structured LLM prompt
3. **LLM** — Llama 3.1 8B Instruct via Ollama, temperature=0
4. **Parser** — extract `VALUE:<answer>` from LLM response with fallback for short raw answers

### Knowledge Base Configurations
- **Clean KB** — correct values only (datasets 2+3+4)
- **Noisy KB** — 50% correct values + 50% randomly injected plausible but wrong values

### Evaluation
- **Rotating query dataset** — each of the 4 datasets used as query in turn, giving ~3,500 evaluation tasks
- **Partial match metric** — prediction correct if normalised value is contained in ground truth or vice versa
- **Numeric tolerance** — prediction correct if within 10% of ground truth for numeric attributes
- **LLM-based evaluation** — LLM judges predictions as `correct`, `acceptable`, or `wrong`

---

## Experiments

| Experiment | Description |
|---|---|
| `experiment2.ipynb` | Pilot: 25 query rows, Dataset 1 only, 3 configs |
| `experiment3.ipynb` | Full eval: all recoverable rows, top-k tuning (1/3/5), partial KB |
| `experiment4.ipynb` | Redesigned: rotating KB, noisy KB, new attributes, LLM eval |
| `error_analysis.ipynb` | Error patterns, retrieval recall analysis |

### Key Results (Experiment 3)

| Config | Overall Accuracy |
|---|---|
| LLM-only | 31.6% |
| RAG-full-k1 | 40.2% |
| RAG-full-k3 | 49.3% |
| RAG-full-k5 | **55.7%** |
| RAG-partial | 50.1% |

---

## Setup

### Requirements
```bash
pip install sentence-transformers pandas numpy requests matplotlib seaborn scikit-learn
```

### Ollama (LLM inference)
```bash
# Install Ollama and pull the model
ollama pull llama3.1:8b
ollama serve &
```

### On bwUniCluster 3.0
```bash
# Request GPU node
salloc -p gpu_a100_il --nodes=1 --ntasks=1 --gres=gpu:1 --time=04:00:00

# Or submit as batch job
sbatch run_experiment4.sh
```

### Local model path (if HuggingFace is unavailable)
```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
LOCAL_MODEL_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)
```

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Ahmad et al. (2023). *RetClean: Retrieval-based Data Cleaning using Foundation Models and Data Lakes*. VLDB.
- Narayan et al. (2022). *Can Foundation Models Wrangle Your Data?* VLDB.