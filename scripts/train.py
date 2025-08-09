"""Main training script for SmallLM."""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from configs.model_config import ModelConfig, TrainingConfig
from model.transformer import GPTModel
from data.tokenizer import create_tokenizer_from_dataset
from data.preprocessing import create_data_loaders
from training.trainer import Trainer
from training.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train SmallLM GPT model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_samples', type=int, default=None, help='Max training samples')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    if args.max_samples:
        print(f"Using max {args.max_samples} training samples")
    
    print("=== Model Configuration ===")
    for key, value in model_config.__dict__.items():
        print(f"{key}: {value}")
    
    print("\n=== Training Configuration ===")
    for key, value in training_config.__dict__.items():
        print(f"{key}: {value}")
    
    # Create or load tokenizer
    tokenizer_path = "tinystories_bpe.model"
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        from data.tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Creating new tokenizer...")
        tokenizer = create_tokenizer_from_dataset(
            dataset_name=training_config.dataset_name,
            vocab_size=model_config.vocab_size,
            max_samples=100000,  # Use subset for tokenizer training
        )
    
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Update model config with actual vocab size
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_name=training_config.dataset_name,
        tokenizer=tokenizer,
        max_length=model_config.max_position_embeddings,
        batch_size=training_config.batch_size,
        max_samples=args.max_samples,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader) if val_loader else 0}")
    
    # Create model
    print("\nInitializing model...")
    model = GPTModel(model_config)
    
    # Calculate estimated batch size and steps
    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
    estimated_steps_per_epoch = len(train_loader) // training_config.gradient_accumulation_steps
    estimated_epochs = training_config.max_steps // estimated_steps_per_epoch
    
    print(f"\nTraining setup:")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Estimated steps per epoch: {estimated_steps_per_epoch}")
    print(f"Estimated epochs: {estimated_epochs}")
    print(f"Max steps: {training_config.max_steps}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        tokenizer_path=tokenizer_path,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.save_checkpoint()
        raise
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()