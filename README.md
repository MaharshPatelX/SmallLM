# SmallLM: GPT-Style Transformer from Scratch

A complete implementation of a GPT-style decoder-only transformer language model trained from scratch on the TinyStories dataset.

## Features

- **Architecture**: GPT-style decoder-only transformer (~20-30M parameters)
- **Dataset**: TinyStories (synthetic short stories from GPT-3.5/GPT-4)
- **Tokenization**: SentencePiece BPE with 16K vocabulary
- **Positional Encoding**: Rotary Positional Embedding (RoPE)
- **Training**: AdamW optimizer with cosine scheduling and warmup
- **Optimizations**: Mixed precision (FP16), gradient checkpointing, gradient accumulation
- **Generation**: Temperature, top-k, top-p sampling strategies

## Model Architecture

| Component | Configuration |
|-----------|---------------|
| **Layers** | 8 transformer blocks |
| **Hidden Size** | 384 dimensions |
| **Attention Heads** | 6 heads (64 dim each) |
| **Feed-Forward** | 1536 dimensions (4x hidden) |
| **Context Length** | 512 tokens |
| **Vocabulary** | 16K tokens (BPE) |
| **Parameters** | ~20-30M total |

## Quick Start

### 1. Setup Environment

Using UV (recommended):
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 2. Download Data and Train Tokenizer

```bash
python scripts/download_data.py
```

This will:
- Download the TinyStories dataset from Hugging Face
- Train a SentencePiece BPE tokenizer with 16K vocabulary
- Save the tokenizer model as `tinystories_bpe.model`

### 3. Train the Model

```bash
python scripts/train.py
```

Optional arguments:
- `--max_samples`: Limit training samples for testing
- `--device`: Device to use (cuda/cpu)
- `--seed`: Random seed for reproducibility

### 4. Generate Text

```bash
python scripts/generate.py checkpoints/checkpoint_best.pt --prompt "Once upon a time"
```

Generation options:
- `--max_tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top_k`: Top-k sampling (default: 50)
- `--top_p`: Top-p sampling (default: 0.9)
- `--num_samples`: Number of samples to generate

Interactive mode:
```bash
python scripts/generate.py interactive checkpoints/checkpoint_best.pt
```

## Project Structure

```
smalllm/
├── configs/           # Model and training configurations
├── data/             # Dataset and tokenization utilities
├── model/            # Transformer architecture implementation
├── training/         # Training loop and optimization
├── inference/        # Text generation utilities
├── scripts/          # Main training and generation scripts
├── tests/            # Unit tests
├── checkpoints/      # Model checkpoints (created during training)
└── README.md
```

## Training Details

### Hyperparameters

- **Learning Rate**: 3e-4 with cosine decay
- **Warmup Steps**: 2000 steps
- **Batch Size**: 256 (32 per device × 8 accumulation steps)
- **Weight Decay**: 0.1 (AdamW)
- **Gradient Clipping**: 1.0
- **Max Steps**: 50000 steps

### Optimizations

- **Mixed Precision**: FP16 training with automatic loss scaling
- **Gradient Checkpointing**: 60% memory reduction, ~20% speed cost
- **Gradient Accumulation**: Simulate larger batch sizes on small GPUs
- **Memory Efficient**: Supports training on 8-12GB VRAM

### Expected Performance

- **Training Time**: 6-12 hours on RTX 3090/4090
- **Memory Usage**: 8-12GB VRAM with optimizations
- **Final Loss**: <2.0 (cross-entropy)
- **Generation Quality**: Coherent short stories with simple vocabulary

## Usage Examples

### Python API

```python
from inference.generator import TextGenerator

# Load trained model
generator = TextGenerator.from_checkpoint(
    checkpoint_path="checkpoints/checkpoint_best.pt",
    tokenizer_path="tinystories_bpe.model"
)

# Generate text
story = generator.generate(
    prompt="Once upon a time, there was a brave little mouse",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
print(story)

# Generate complete story
story = generator.generate_story(
    theme="friendship",
    max_length=200
)
```

### Configuration

Modify `configs/model_config.py` to change model architecture:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 16000
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    max_position_embeddings: int = 512
    # ... other parameters
```

## Testing

Run unit tests:
```bash
python tests/test_model.py
```

Test individual components:
```bash
python -c "
from model.transformer import GPTModel
from configs.model_config import ModelConfig
model = GPTModel(ModelConfig())
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

## Deployment on Vast.ai

1. Connect to GPU instance:
```bash
ssh -p 19726 root@185.150.27.254 -L 8080:localhost:8080
```

2. Setup environment:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone repository
git clone https://github.com/MaharshPatelX/smalllm.git
cd smalllm

# Install dependencies
uv pip install -r requirements.txt
```

3. Run training:
```bash
python scripts/train.py --device cuda
```

## Monitoring

The trainer supports Weights & Biases logging:

```python
# Enable in TrainingConfig
use_wandb: bool = True
project_name: str = "smalllm-gpt"
```

## Troubleshooting

### Memory Issues
- Reduce batch size or increase gradient accumulation steps
- Enable gradient checkpointing (default: enabled)
- Use mixed precision training (default: enabled)

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Increase batch size if memory allows
- Ensure data loading is efficient (check num_workers)

### Poor Generation Quality
- Train for more steps
- Adjust temperature and sampling parameters
- Check training loss convergence
- Verify tokenizer quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)

## Acknowledgments

- Hugging Face for the TinyStories dataset
- The PyTorch team for the excellent framework
- Andrej Karpathy for educational transformer implementations