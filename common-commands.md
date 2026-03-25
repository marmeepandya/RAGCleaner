# Python Environment
source /home/ma/ma_ma/ma_mpandya/CAC/.venv/bin/activate

# Allocating GPU
salloc -p gpu_a100_il --nodes=1 --ntasks=1 --gres=gpu:1 --time=04:00:00

# Checking the process sequence
squeue -u ma_mpandya

# Start time of the GPU allocation
squeue --start -u ma_mpandya

# Ollama
module load cs/ollama/0.5.11
ollama serve &
sleep 10