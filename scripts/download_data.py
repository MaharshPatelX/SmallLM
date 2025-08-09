"""Download and prepare TinyStories dataset."""

import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset
from data.tokenizer import create_tokenizer_from_dataset


def download_dataset(dataset_name: str = "roneneldan/TinyStories", cache_dir: str = None):
    """Download TinyStories dataset."""
    print(f"Downloading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    print("Dataset info:")
    print(f"Train samples: {len(dataset['train'])}")
    
    # Show sample texts
    print("\nSample texts:")
    for i in range(3):
        text = dataset['train'][i]['text']
        print(f"\n--- Sample {i+1} ---")
        print(text[:200] + "..." if len(text) > 200 else text)
    
    return dataset


def create_tokenizer(
    dataset_name: str = "roneneldan/TinyStories",
    vocab_size: int = 16000,
    max_samples: int = 100000,
):
    """Create and train SentencePiece tokenizer."""
    print(f"\nTraining tokenizer...")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max samples for training: {max_samples}")
    
    tokenizer = create_tokenizer_from_dataset(
        dataset_name=dataset_name,
        vocab_size=vocab_size,
        max_samples=max_samples,
    )
    
    # Test tokenizer
    test_text = "Once upon a time, there was a little girl who loved to play with her toys."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTokenizer test:")
    print(f"Original: {test_text}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Decoded: {decoded}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Download TinyStories dataset and create tokenizer')
    parser.add_argument('--dataset', type=str, default='roneneldan/TinyStories', help='Dataset name')
    parser.add_argument('--vocab_size', type=int, default=16000, help='Tokenizer vocabulary size')
    parser.add_argument('--max_samples', type=int, default=100000, help='Max samples for tokenizer training')
    parser.add_argument('--cache_dir', type=str, help='Cache directory for dataset')
    parser.add_argument('--skip_download', action='store_true', help='Skip dataset download')
    parser.add_argument('--skip_tokenizer', action='store_true', help='Skip tokenizer creation')
    
    args = parser.parse_args()
    
    # Download dataset
    if not args.skip_download:
        dataset = download_dataset(args.dataset, args.cache_dir)
        print(f"Dataset downloaded and cached")
    
    # Create tokenizer
    if not args.skip_tokenizer:
        tokenizer = create_tokenizer(
            dataset_name=args.dataset,
            vocab_size=args.vocab_size,
            max_samples=args.max_samples,
        )
        print(f"Tokenizer created and saved")
    
    print("Setup complete!")


if __name__ == "__main__":
    main()