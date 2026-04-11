# bwUniCluster 3.0 — Common Commands
## RAG Data Cleaning Project Reference

---

## 1. Python Environments

```bash
# RAG project venv (use this for experiments)
source /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/bin/activate

# CAC project venv
source /home/ma/ma_ma/ma_mpandya/CAC/.venv/bin/activate

# Deactivate any active venv
deactivate

# Check which Python / venv is active
which python
```

---

## 2. Connecting to the Cluster

```bash
# Login (from your Mac)
ssh ma_mpandya@uc3.scc.kit.edu

# Check which node you are on
hostname
# uc3n990 or uc3n991 = LOGIN NODE (no GPU, do not run experiments here)
# uc2n9xx = COMPUTE NODE (has GPU)
```

> **Important:** Never run Ollama or heavy computations on the login node.
> Always use a compute node via `salloc` or `sbatch`.

---

## 3. GPU Compute Node — Interactive Session

```bash
# Request an A100 GPU node interactively
salloc -p gpu_a100_il --nodes=1 --ntasks=1 --gres=gpu:1 --time=12:00:00

# For shorter jobs
salloc -p cpu -n 1 -t 120 --mem=5000

# After it is granted, you will be on uc2n9xx automatically
# Confirm with:
hostname

# Check available partitions and idle nodes
sinfo_t_idle
```

> **Note:** You cannot SSH directly into a compute node (it requires OTP).
> Use `salloc` for interactive sessions or `sbatch` for batch jobs.

---

## 4. GPU Compute Node — Batch Job (recommended for long runs)

```bash
# Submit a batch job
sbatch run_experiment.sh

# Check job status
squeue -u ma_mpandya
# ST column: PD = pending, R = running, CG = completing/cancelling

# Check estimated start time
squeue --start -u ma_mpandya

# Monitor job output live (once it starts running)
tail -f rag_full_<JOBID>.out
tail -f rag_full_<JOBID>.err

# Cancel a job
scancel <JOBID>

# Available Partition
sinfo_t_idlw

# Check job history (after it finishes)
sacct -u ma_mpandya --format=JobID,JobName,State,ExitCode,Start,End --starttime=today
```

### Template batch script

```bash
cat > run_experiment.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rag_exp
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/exp_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/exp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

module load cs/ollama/0.5.11
ollama serve &
sleep 15

source /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/bin/activate
cd /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning

jupyter nbconvert --to script my_experiment.ipynb
python my_experiment.py

pkill ollama
EOF
```

---

## 5. Ollama

```bash
# Load the Ollama module (must do this before any ollama command)
module load cs/ollama/0.5.11

# Start Ollama server in the background
ollama serve &
sleep 15

# Your full startup sequence should now be:
module load cs/ollama/0.5.11
module load devel/cuda/12.8
OLLAMA_HOST=127.0.0.1:11435 ollama serve &
sleep 15
OLLAMA_HOST=127.0.0.1:11435 ollama ps

# Check what is loaded and whether it is using GPU
ollama ps
# Good output: llama3.1:8b  ...  100% GPU  NVIDIA A100
# Bad output:  llama3.1:8b  ...  100% CPU  (running on CPU, too slow)

# List available models
ollama list

# Kill Ollama
pkill ollama

https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture

# Test if Ollama is responding (and measure speed)
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"Say OK","stream":false}' | \
  python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['response']); print('Duration (ns):', r.get('total_duration','?'))"
# Under 2,000,000,000 ns (2 seconds) = GPU
# Over 40,000,000,000 ns (40 seconds) = CPU
```

> **Why GPU matters:** On CPU, each LLM call takes 40-50 seconds.
> On A100 GPU, each call takes 1-2 seconds. With 900 tasks, that is
> the difference between 12 hours and 30 minutes.

---

## 6. Sentence Transformer (HuggingFace offline)

```bash
# The model is cached here — no internet needed
ls ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/
# Should show: c9745ed1d9f207416be6d2e6f8de32d1f16199bf
```

In your Python code, always load it like this:

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LOCAL_MODEL_PATH = "/home/ma/ma_ma/ma_mpandya/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)
```

---

## 7. sys.path Fix for Notebooks

Always add these at the top of every notebook cell 1:

```python
import sys, os
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib64/python3.12/site-packages')
sys.path.insert(0, '/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/lib/python3.12/site-packages')
sys.path.append('/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI')
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

---

## 8. Git

```bash
# Push to GitHub (must use token in URL — token expired, generate a new one at github.com)
git remote set-url origin https://YOUR_TOKEN@github.com/marmeepandya/RAGCleaner.git

# Standard workflow
git add .
git commit -m "your message"
git push origin main

# Check status
git status
git log --oneline -5
```

---

## 9. Common Problem Fixes

| Problem | Fix |
|---|---|
| `No module named 'pandas'` | Wrong venv selected — run `source .../venv/bin/activate` |
| `no compatible GPUs were discovered` | You are on login node — use `salloc` or `sbatch` |
| `Connection refused` on port 11434 | Ollama not running — run `module load cs/ollama/0.5.11 && ollama serve &` |
| `ollama: command not found` | Module not loaded — run `module load cs/ollama/0.5.11` |
| LLM taking 40+ seconds per call | Running on CPU not GPU — check `ollama ps` |
| Notebook kernel stuck | Run `pkill -f ipykernel` in terminal, then restart kernel |
| `jupyter nbconvert not found` | Use the RAG project venv, not the system Python |
| HuggingFace 401 Unauthorized | Token expired — set `TRANSFORMERS_OFFLINE=1` and use local model path |

---

## 10. File Locations

```
~/RAG_Data_Cleaning/
├── PyDI/PyDI/cleaners/rag_cleaner.py     # RAGCleaner implementation
├── normalized_products/                   # Input datasets (JSON)
├── experiment8.2_half_kb_easy.ipynb       # Current experiment
├── results_*.csv                          # Experiment results
├── fig_*.png                              # Generated figures
├── report.tex                             # Seminar report
├── references.bib                         # Bibliography
├── run_rag_full.sh                        # SLURM batch script
└── requirements.txt                       # Python dependencies
```