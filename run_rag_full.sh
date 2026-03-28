#!/bin/bash
#SBATCH --job-name=rag_full
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/rag_full_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/rag_full_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

module load cs/ollama/0.5.11
ollama serve &
sleep 15
ollama ps

source /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/bin/activate
cd /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning

jupyter nbconvert --to script experiment8.2_half_kb_easy.ipynb
python experiment8.2_half_kb_easy.py

pkill ollama
