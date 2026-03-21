#!/bin/bash
#SBATCH --job-name=rag_exp4
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/exp4_%j.out
#SBATCH --error=/home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/exp4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marmeep23@gmail.com

module load cs/ollama/0.5.11
ollama serve &
sleep 15

source /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning/PyDI/venv/bin/activate
cd /home/ma/ma_ma/ma_mpandya/RAG_Data_Cleaning

python experiment4_simple.py

pkill ollama
