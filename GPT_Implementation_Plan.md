# GPT-Style Transformer from Scratch - Implementation Plan

## Project Overview

Build a complete GPT-style decoder-only transformer language model trained from scratch using PyTorch with ~20-30M parameters, capable of generating coherent short stories using the TinyStories dataset.

## 📋 Project Structure

```
smalllm/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # TinyStories dataset loader
│   ├── tokenizer.py        # SentencePiece BPE tokenizer
│   └── preprocessing.py    # Data preprocessing utilities
├── model/
│   ├── __init__.py
│   ├── transformer.py      # Main GPT model
│   ├── attention.py        # Multi-head attention with RoPE
│   ├── layers.py          # Feed-forward, embeddings, layer norm
│   └── positional.py      # Rotary positional encoding
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Training loop with checkpointing
│   ├── optimizer.py       # AdamW + cosine scheduler setup
│   └── utils.py          # Training utilities
├── inference/
│   ├── __init__.py
│   └── generator.py       # Text generation script
├── tests/
│   ├── test_model.py
│   ├── test_data.py
│   └── test_training.py
├── scripts/
│   ├── train.py          # Main training script
│   ├── generate.py       # Generation script
│   └── download_data.py  # Data download script
├── configs/
│   └── model_config.py   # Model hyperparameters
├── requirements.txt
├── pyproject.toml        # UV configuration
└── README.md
```

## 🎯 Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Parameters** | ~20-30M | Total trainable parameters |
| **Layers** | 8 | Number of transformer blocks |
| **Hidden Dimension** | 384 | Model embedding dimension |
| **Attention Heads** | 6 | Multi-head attention heads |
| **Context Length** | 512 | Maximum sequence length |
| **Vocabulary Size** | 16K | BPE tokenizer vocabulary |
| **Architecture** | Decoder-only | GPT-style transformer |
| **Activation** | GELU | Activation function |
| **Normalization** | LayerNorm (pre-norm) | Normalization strategy |
| **Positional Encoding** | RoPE | Rotary Positional Embedding |

## 📊 Dataset Information

### TinyStories Dataset
- **Source**: `roneneldan/TinyStories` on Hugging Face
- **Description**: Synthetically generated short stories by GPT-3.5/GPT-4
- **Paper**: https://arxiv.org/abs/2305.07759
- **Size**: ~2.2M stories
- **Vocabulary**: Simple words suitable for small models
- **License**: cdla-sharing-1.0

### Available Files
- `TinyStories-train.txt` - Main training data
- `TinyStoriesV2-GPT4-train.txt` - GPT-4 only version (higher quality)
- `tinystories-valid.txt` - Validation data
- `tinystories_all_data.tar.gz` - Complete archive with metadata

### Download Command
```python
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories")
```

## 🏗️ Architecture Implementation

### Core Components

#### 1. Multi-Head Attention with RoPE
```python
# Key features:
- Rotary Positional Encoding integration
- Causal masking for autoregressive generation
- Scaled dot-product attention
- Flash attention optimization (optional)
```

#### 2. Feed-Forward Networks
```python
# Configuration:
- Hidden dimension: 384 * 4 = 1536
- Activation: GELU
- Dropout for regularization
```

#### 3. Layer Normalization
```python
# Pre-norm configuration:
- Applied before attention and FFN
- Improves training stability
- Better gradient flow
```

#### 4. Embeddings
```python
# Components:
- Token embeddings: vocab_size × hidden_dim
- Learnable parameters
- Shared with output projection (optional)
```

### Rotary Positional Encoding (RoPE)

#### Implementation Options
1. **TorchTune**: `torchtune.modules.RotaryPositionalEmbeddings`
2. **lucidrains**: `rotary-embedding-torch` library
3. **Custom**: Educational implementation

#### Key Benefits
- Better length extrapolation
- Relative position encoding
- No additional parameters
- Improved performance on long sequences

```python
# Basic usage:
rotary_emb = RotaryEmbedding(dim=head_dim)
q = rotary_emb.rotate_queries_or_keys(q)
k = rotary_emb.rotate_queries_or_keys(k)
```

## 🚀 Training Configuration

### Optimizer: AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    eps=1e-8
)
```

### Learning Rate Schedule: Cosine with Warmup
```python
# Configuration:
- Warmup steps: 2000
- Total steps: calculated from epochs and batch size
- Min LR: 10% of max LR
- Cosine annealing after warmup
```

### Training Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Learning Rate** | 3e-4 | Initial learning rate |
| **Batch Size** | 256 | With gradient accumulation if needed |
| **Weight Decay** | 0.1 | L2 regularization |
| **Dropout** | 0.1 | Regularization |
| **Gradient Clipping** | 1.0 | Prevent gradient explosion |
| **Mixed Precision** | FP16 | Memory and speed optimization |
| **Warmup Steps** | 2000 | Learning rate warmup |
| **Max Epochs** | 10-20 | Depending on convergence |

## 📥 Data Pipeline

### 1. Dataset Download
```python
from datasets import load_dataset

# Download TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories")
train_data = dataset['train']['text']
```

### 2. Tokenization with SentencePiece
```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='train_data.txt',
    model_prefix='tinystories_bpe',
    vocab_size=16000,
    model_type='bpe',
    character_coverage=1.0,
    split_by_unicode_script=True,
    split_by_number=True,
    split_by_whitespace=True
)
```

### 3. Data Loading
```python
class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        # Padding and truncation logic
        return torch.tensor(tokens)
```

## 🔧 Memory Optimization

### Gradient Checkpointing
```python
# Benefits:
- 60% memory reduction
- 20% increased training time
- Enable larger batch sizes or models

# Implementation:
torch.utils.checkpoint.checkpoint(layer, x)
```

### Mixed Precision Training
```python
# Using torch.amp
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation
```python
# Simulate larger batch sizes
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📈 Implementation Phases

### Phase 1: Environment & Data Setup
- [x] Research dataset and architecture
- [ ] Set up UV project with dependencies
- [ ] Download and explore TinyStories dataset
- [ ] Train SentencePiece tokenizer
- [ ] Create data loading pipeline

### Phase 2: Model Architecture
- [ ] Implement RoPE positional encoding
- [ ] Build multi-head attention mechanism
- [ ] Create feed-forward and normalization layers
- [ ] Assemble complete GPT model
- [ ] Add gradient checkpointing
- [ ] Test forward pass and count parameters

### Phase 3: Training Infrastructure
- [ ] Set up AdamW optimizer with cosine scheduling
- [ ] Implement training loop with mixed precision
- [ ] Add logging (wandb/tensorboard)
- [ ] Implement checkpointing and model saving
- [ ] Create validation loop

### Phase 4: Training & Validation
- [ ] Run initial training experiments
- [ ] Monitor loss curves and stability
- [ ] Tune hyperparameters if needed
- [ ] Implement early stopping
- [ ] Save best model checkpoints

### Phase 5: Inference & Generation
- [ ] Implement text generation pipeline
- [ ] Add sampling strategies (temperature, top-k, top-p)
- [ ] Create inference scripts
- [ ] Test generation quality
- [ ] Optimize inference speed

### Phase 6: Testing & Documentation
- [ ] Write comprehensive unit tests
- [ ] Performance benchmarks
- [ ] Memory profiling
- [ ] Create usage examples
- [ ] Documentation and README

## 🛠️ Dependencies

### Core Libraries
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.0
sentencepiece>=0.1.99
```

### Training & Utilities
```txt
wandb>=0.15.0
tqdm>=4.65.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Development
```txt
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
```

### UV Configuration (pyproject.toml)
```toml
[project]
name = "smalllm"
version = "0.1.0"
description = "GPT-style transformer from scratch"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "sentencepiece>=0.1.99",
    "wandb>=0.15.0",
    "tqdm>=4.65.0"
]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

## 💾 Expected Performance

### Training Metrics
| Metric | Expected Value | Notes |
|--------|----------------|-------|
| **Training Time** | 6-12 hours | Single GPU (RTX 3090/4090) |
| **Final Loss** | <2.0 | Cross-entropy loss |
| **Memory Usage** | 8-12GB VRAM | With optimizations |
| **Parameters** | 20-30M | Actual count may vary |

### Generation Quality
- Coherent short stories
- Simple vocabulary usage
- Basic narrative structure
- Consistent character names
- Appropriate story length (50-200 words)

## 🔗 Key Resources

### Dataset
- **TinyStories**: https://huggingface.co/datasets/roneneldan/TinyStories
- **Paper**: https://arxiv.org/abs/2305.07759

### Architecture References
- **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- **RoPE Paper**: https://arxiv.org/abs/2104.09864
- **GPT Paper**: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

### Implementation References
- **minGPT**: https://github.com/karpathy/minGPT
- **Rotary Embedding**: https://github.com/lucidrains/rotary-embedding-torch
- **PyTorch Tutorial**: https://pytorch.org/tutorials/

### Training Resources
- **Mixed Precision**: https://pytorch.org/docs/stable/amp.html
- **Gradient Checkpointing**: https://pytorch.org/docs/stable/checkpoint.html
- **Hugging Face Optimizers**: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

## 🎯 Success Criteria

### Model Performance
- [ ] Model trains without memory issues
- [ ] Loss decreases consistently during training
- [ ] No gradient explosion or vanishing
- [ ] Model generates coherent text
- [ ] Validation loss follows training loss

### Code Quality
- [ ] All components are thoroughly tested
- [ ] Code follows best practices
- [ ] Comprehensive documentation
- [ ] Reproducible results
- [ ] Efficient memory usage

### Deliverables
- [ ] Complete working transformer implementation
- [ ] Trained model checkpoint
- [ ] Generation examples
- [ ] Training logs and metrics
- [ ] Documentation and usage guide

## 📝 Notes

### Training Tips
- Start with smaller model for debugging
- Monitor gradient norms during training
- Use learning rate finder if needed
- Implement proper validation split
- Save model checkpoints frequently

### Common Issues
- Memory overflow: Reduce batch size or use gradient accumulation
- Slow convergence: Check learning rate and warmup
- Poor generation: Verify model architecture and training data
- Gradient explosion: Implement gradient clipping

### Future Improvements
- Implement Flash Attention for efficiency
- Add support for longer sequences
- Experiment with different positional encodings
- Try different sampling strategies
- Implement model parallelism for larger models

---

**Project Goal**: Build a complete, working GPT-style transformer that demonstrates understanding of modern NLP architecture and training techniques, capable of generating coherent short stories after training on the TinyStories dataset.