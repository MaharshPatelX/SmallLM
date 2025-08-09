"""Test model components."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytest
from configs.model_config import ModelConfig
from model.transformer import GPTModel, TransformerBlock
from model.attention import MultiHeadAttention
from model.layers import MLP, LayerNorm
from model.positional import RotaryPositionalEncoding


def test_model_config():
    """Test model configuration."""
    config = ModelConfig()
    
    assert config.vocab_size > 0
    assert config.hidden_size % config.num_heads == 0
    assert config.num_layers > 0
    assert config.intermediate_size > 0


def test_layer_norm():
    """Test LayerNorm."""
    hidden_size = 384
    layer_norm = LayerNorm(hidden_size)
    
    x = torch.randn(2, 10, hidden_size)
    output = layer_norm(x)
    
    assert output.shape == x.shape
    assert torch.allclose(output.mean(dim=-1), torch.zeros(2, 10), atol=1e-5)
    assert torch.allclose(output.std(dim=-1), torch.ones(2, 10), atol=1e-5)


def test_mlp():
    """Test MLP layer."""
    hidden_size = 384
    intermediate_size = 1536
    
    mlp = MLP(hidden_size, intermediate_size)
    
    x = torch.randn(2, 10, hidden_size)
    output = mlp(x)
    
    assert output.shape == x.shape


def test_rotary_positional_encoding():
    """Test RoPE."""
    dim = 64
    seq_len = 128
    
    rope = RotaryPositionalEncoding(dim)
    
    q = torch.randn(2, 8, seq_len, dim)
    k = torch.randn(2, 8, seq_len, dim)
    
    q_rotated, k_rotated = rope(q, k)
    
    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape


def test_multi_head_attention():
    """Test multi-head attention."""
    config = ModelConfig()
    
    attention = MultiHeadAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        use_rope=config.use_rope
    )
    
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = attention(x)
    
    assert output.shape == x.shape


def test_transformer_block():
    """Test transformer block."""
    config = ModelConfig()
    
    block = TransformerBlock(config)
    
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = block(x)
    
    assert output.shape == x.shape


def test_gpt_model():
    """Test complete GPT model."""
    config = ModelConfig()
    config.vocab_size = 1000  # Smaller vocab for testing
    config.num_layers = 4     # Fewer layers for testing
    
    model = GPTModel(config)
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids=input_ids, labels=labels)
    
    assert 'loss' in outputs
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    assert outputs['loss'].item() > 0


def test_model_generation():
    """Test text generation."""
    config = ModelConfig()
    config.vocab_size = 1000
    config.num_layers = 4
    
    model = GPTModel(config)
    model.eval()
    
    batch_size, seq_len = 1, 10
    input_ids = torch.randint(1, config.vocab_size-1, (batch_size, seq_len))
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=1.0,
            do_sample=True
        )
    
    assert generated.shape[0] == batch_size
    assert generated.shape[1] > seq_len


def test_model_parameter_count():
    """Test parameter counting."""
    config = ModelConfig()
    
    model = GPTModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check if within expected range (20-30M)
    assert 15_000_000 < total_params < 35_000_000
    assert total_params == trainable_params


if __name__ == "__main__":
    # Run tests
    test_model_config()
    test_layer_norm()
    test_mlp()
    test_rotary_positional_encoding()
    test_multi_head_attention()
    test_transformer_block()
    test_gpt_model()
    test_model_generation()
    test_model_parameter_count()
    
    print("All model tests passed!")