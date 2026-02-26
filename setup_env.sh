#!/bin/bash
set -e

ENV_NAME="gpt2_gpipe_mem"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.10 -y

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing transformers ==="
pip install transformers datasets

echo "=== Done! ==="
echo "To run:  conda activate $ENV_NAME && python gpt2_gpipe_memory_profile.py"
