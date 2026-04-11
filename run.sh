#!/bin/bash
#SBATCH --job-name=reranker_fullKB
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/3.reranker_fullKB_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/3.reranker_fullKB_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

module load cs/ollama/0.5.11
module load devel/cuda/12.8

OLLAMA_HOST=127.0.0.1:11435 ollama serve &
sleep 15
OLLAMA_HOST=127.0.0.1:11435 ollama ps

source /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/bin/activate
cd /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning


python '3.reranker_fullKB.py'

pkill ollama