"""Text generation script for SmallLM."""

import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from configs.model_config import ModelConfig
from model.transformer import GPTModel
from data.tokenizer import SentencePieceTokenizer


def load_model_and_tokenizer(checkpoint_path: str, device: str = 'cuda'):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    config = checkpoint.get('config')
    if config is None:
        # Use default config if not saved in checkpoint
        config = ModelConfig()
        print("Warning: Config not found in checkpoint, using default")
    else:
        # Convert dict to ModelConfig if needed
        if isinstance(config, dict):
            model_config = ModelConfig()
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            config = model_config
    
    # Load tokenizer
    tokenizer_path = checkpoint.get('tokenizer_path', 'tinystories_bpe.model')
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
    
    # Update config with actual vocab size
    config.vocab_size = tokenizer.vocab_size
    
    # Create and load model
    model = GPTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return model, tokenizer, config


def generate_text(
    model: GPTModel,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    do_sample: bool = True,
    device: str = 'cuda'
):
    """Generate text from prompt."""
    
    # Encode prompt
    if prompt.strip():
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    else:
        # Start with BOS token if no prompt
        input_ids = [tokenizer.bos_id]
    
    input_ids = torch.tensor([input_ids], device=device)
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_id,
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    
    # Remove BOS/EOS tokens from display
    generated_text = generated_text.replace('<s>', '').replace('</s>', '').strip()
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with SmallLM')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--no_sample', action='store_true', help='Use greedy decoding')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)
    
    print(f"\nGeneration settings:")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Sampling: {not args.no_sample}")
    print(f"Device: {device}")
    print()
    
    # Generate samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"=== Sample {i+1} ===")
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.no_sample,
            device=device
        )
        
        print("Generated text:")
        print(generated_text)
        
        if args.num_samples > 1:
            print()


def interactive_mode():
    """Interactive generation mode."""
    parser = argparse.ArgumentParser(description='Interactive text generation')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)
    
    print("Interactive mode - Type 'quit' to exit")
    print("Commands: !temp X, !topk X, !topp X, !tokens X")
    
    temperature = 1.0
    top_k = 50
    top_p = 0.9
    max_tokens = 100
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
            
            # Handle commands
            if prompt.startswith('!'):
                parts = prompt.split()
                cmd = parts[0][1:]
                if len(parts) > 1:
                    value = parts[1]
                    if cmd == 'temp':
                        temperature = float(value)
                        print(f"Temperature set to {temperature}")
                    elif cmd == 'topk':
                        top_k = int(value)
                        print(f"Top-k set to {top_k}")
                    elif cmd == 'topp':
                        top_p = float(value)
                        print(f"Top-p set to {top_p}")
                    elif cmd == 'tokens':
                        max_tokens = int(value)
                        print(f"Max tokens set to {max_tokens}")
                continue
            
            # Generate
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            
            print("Generated:")
            print(generated_text)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Remove 'interactive' from args
        sys.argv.pop(1)
        interactive_mode()
    else:
        main()