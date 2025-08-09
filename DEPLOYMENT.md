# Deployment Instructions for Vast.ai

## Quick Start on Vast.ai GPU Instance

### 1. Connect to Your Vast.ai Instance

```bash
ssh -p 19726 root@185.150.27.254 -L 8080:localhost:8080
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/MaharshPatelX/SmallLM.git
cd SmallLM

# Make deployment script executable
chmod +x deploy_vast.sh

# Run deployment (installs everything and tests)
./deploy_vast.sh
```

### 3. Full Training

```bash
# Start full training
python3 scripts/train.py

# Or with custom settings
python3 scripts/train.py --max_samples 500000 --seed 42
```

### 4. Monitor Training (Optional)

Set up Weights & Biases monitoring:
```bash
# Install and setup wandb
pip install wandb
wandb login  # Enter your API key

# Training will automatically log to wandb
python3 scripts/train.py
```

### 5. Generate Text

```bash
# Generate text from trained model
python3 scripts/generate.py checkpoints/checkpoint_best.pt \
    --prompt "Once upon a time, there was a brave little mouse" \
    --max_tokens 150 \
    --temperature 0.8 \
    --num_samples 3

# Interactive generation mode
python3 scripts/generate.py interactive checkpoints/checkpoint_best.pt
```

## Manual Installation Steps

If the automatic deployment script doesn't work:

### Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Install Dependencies
```bash
# Core PyTorch with CUDA support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Additional dependencies
uv pip install transformers datasets sentencepiece tokenizers wandb tqdm numpy matplotlib seaborn accelerate rotary-embedding-torch
```

### Prepare Data
```bash
# Download dataset and train tokenizer
python3 scripts/download_data.py --max_samples 100000
```

### Test Installation
```bash
python3 test_basic.py
```

## Expected Performance on RTX 4090

- **Memory Usage**: 8-12GB VRAM
- **Training Speed**: ~1000-1500 tokens/second
- **Training Time**: 6-12 hours for full training
- **Dataset Size**: ~2.2M stories (~600MB)
- **Final Model Size**: ~100MB

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"
```

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Enable gradient checkpointing (default: on)
- Use gradient accumulation (default: 8 steps)

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase batch size if memory allows
- Check data loading efficiency

### Connection Issues
If SSH connection drops:
```bash
# Use screen or tmux for persistent sessions
screen -S training
python3 scripts/train.py
# Ctrl+A, then D to detach
# screen -r training  # to reattach
```

## File Structure After Setup

```
SmallLM/
├── checkpoints/           # Model checkpoints (created during training)
├── configs/              # Model configuration
├── data/                # Dataset utilities  
├── model/               # Transformer implementation
├── training/            # Training pipeline
├── scripts/             # Main scripts
├── inference/           # Text generation
├── tests/               # Unit tests
├── tinystories_bpe.model # Trained tokenizer
└── deploy_vast.sh       # Deployment script
```

## Key Commands Reference

```bash
# Full training
python3 scripts/train.py

# Training with custom settings
python3 scripts/train.py --max_samples 1000000 --device cuda

# Generate text
python3 scripts/generate.py checkpoints/checkpoint_best.pt --prompt "Hello"

# Interactive generation
python3 scripts/generate.py interactive checkpoints/checkpoint_best.pt

# Test model
python3 test_basic.py

# Download fresh data
python3 scripts/download_data.py
```

## Success Indicators

✅ **Setup Complete**: `./deploy_vast.sh` runs without errors  
✅ **Model Works**: `python3 test_basic.py` passes all tests  
✅ **Training Started**: Loss decreases consistently  
✅ **Generation Works**: Model produces coherent text  

The complete implementation is ready for training on your RTX 4090!