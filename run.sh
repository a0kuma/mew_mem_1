#!/bin/bash
set -e

ENV_NAME="gpt2_gpipe_mem"

echo "============================================"
echo " GPT-2 GPipe Memory Profiling - Runner"
echo "============================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[INFO] Python: $(which python)"
echo "[INFO] PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[INFO] GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Run the profiling script
cd "$(dirname "$0")"
python gpt2_gpipe_memory_profile.py 2>&1 | tee memory_reports/run_log.txt

echo ""
echo "[INFO] Full log saved to memory_reports/run_log.txt"
