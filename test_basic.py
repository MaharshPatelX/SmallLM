#!/usr/bin/env python3
"""Basic test script to verify implementation."""

import sys
from pathlib import Path
sys.path.append('.')

try:
    import torch
    print("✓ PyTorch available")
except ImportError:
    print("✗ PyTorch not available - install requirements first")
    sys.exit(1)

try:
    from configs.model_config import ModelConfig
    from model.transformer import GPTModel
    print("✓ Model imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_model():
    """Test basic model functionality."""
    print("\nTesting model functionality...")
    
    # Create small config for testing
    config = ModelConfig()
    config.vocab_size = 1000
    config.num_layers = 4
    config.hidden_size = 256
    config.num_heads = 4
    config.intermediate_size = 1024
    
    # Create model
    model = GPTModel(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {params:,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
        
        print(f"Forward pass successful")
        print(f"Loss: {loss.item():.4f}")
        print(f"Logits shape: {logits.shape}")
        
        # Test generation
        generated = model.generate(
            input_ids=input_ids[:1, :10],
            max_new_tokens=20,
            temperature=1.0,
            do_sample=True
        )
        print(f"Generation successful - Generated shape: {generated.shape}")
    
    print("✓ All basic tests passed!")

if __name__ == "__main__":
    test_model()