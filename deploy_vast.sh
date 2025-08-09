#!/bin/bash
# Deployment script for Vast.ai GPU instance

set -e

echo "=== SmallLM Deployment Script for Vast.ai ==="

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "✓ UV installed"
else
    echo "✓ UV already available"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers datasets sentencepiece tokenizers wandb tqdm numpy matplotlib seaborn accelerate rotary-embedding-torch

echo "✓ Dependencies installed"

# Test basic functionality
echo "Testing basic functionality..."
python3 test_basic.py

# Download and prepare data
echo "Downloading TinyStories dataset and creating tokenizer..."
python3 scripts/download_data.py --max_samples 50000

echo "✓ Data preparation complete"

# Test training for a few steps
echo "Testing training pipeline..."
python3 scripts/train.py --max_samples 1000 --seed 42

echo "✓ Training test complete"

echo "=== Deployment successful! ==="
echo ""
echo "To start full training, run:"
echo "python3 scripts/train.py"
echo ""
echo "To generate text, run:"
echo "python3 scripts/generate.py checkpoints/checkpoint_best.pt --prompt 'Once upon a time'"
echo ""
echo "To monitor with wandb, set WANDB_API_KEY environment variable"