"""Text generation utilities."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from ..model.transformer import GPTModel
from ..data.tokenizer import SentencePieceTokenizer


class TextGenerator:
    """High-level text generation interface."""
    
    def __init__(
        self,
        model: GPTModel,
        tokenizer: SentencePieceTokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = 'cuda'
    ) -> "TextGenerator":
        """Load generator from checkpoint."""
        from ..configs.model_config import ModelConfig
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load tokenizer
        tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
        
        # Create model config
        config = checkpoint.get('config')
        if config is None:
            config = ModelConfig()
        elif isinstance(config, dict):
            model_config = ModelConfig()
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            config = model_config
        
        config.vocab_size = tokenizer.vocab_size
        
        # Create and load model
        model = GPTModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
        seed: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """Generate text from prompt."""
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encode prompt
        if prompt.strip():
            input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        else:
            input_ids = [self.tokenizer.bos_id]
        
        input_ids = torch.tensor([input_ids] * num_return_sequences, device=self.device)
        
        # Generate
        with torch.no_grad():
            generated_sequences = []
            
            for i in range(num_return_sequences):
                generated_ids = self._generate_sequence(
                    input_ids[i:i+1],
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    do_sample,
                    repetition_penalty,
                )
                generated_sequences.append(generated_ids[0])
        
        # Decode generated sequences
        generated_texts = []
        for generated_ids in generated_sequences:
            text = self.tokenizer.decode(generated_ids.cpu().tolist())
            # Clean up special tokens
            text = text.replace('<s>', '').replace('</s>', '').strip()
            generated_texts.append(text)
        
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Generate a single sequence."""
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                model_input = generated_ids
            else:
                model_input = generated_ids[:, -1:]
            
            outputs = self.model(
                input_ids=model_input,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            logits = outputs['logits']
            past_key_values = outputs['past_key_values']
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty
                )
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample or select next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_id:
                break
        
        return generated_ids
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for token_id in set(generated_ids[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if k <= 0:
            return logits
        
        top_k = min(k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def complete_story(
        self,
        beginning: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Complete a story given a beginning."""
        return self.generate(
            prompt=beginning,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    def generate_story(
        self,
        theme: Optional[str] = None,
        max_length: int = 200,
        temperature: float = 0.8,
    ) -> str:
        """Generate a complete story."""
        if theme:
            prompt = f"Once upon a time, there was a story about {theme}."
        else:
            prompt = "Once upon a time,"
        
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
        )
    
    def interactive_generation(self):
        """Interactive text generation."""
        print("Interactive Generation Mode")
        print("Commands: !temp X, !topk X, !topp X, !tokens X, !quit")
        print("-" * 50)
        
        temperature = 1.0
        top_k = 50
        top_p = 0.9
        max_tokens = 100
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt == '!quit':
                    break
                
                # Handle commands
                if prompt.startswith('!'):
                    parts = prompt.split()
                    cmd = parts[0][1:]
                    if len(parts) > 1:
                        try:
                            value = parts[1]
                            if cmd == 'temp':
                                temperature = float(value)
                                print(f"Temperature: {temperature}")
                            elif cmd == 'topk':
                                top_k = int(value)
                                print(f"Top-k: {top_k}")
                            elif cmd == 'topp':
                                top_p = float(value)
                                print(f"Top-p: {top_p}")
                            elif cmd == 'tokens':
                                max_tokens = int(value)
                                print(f"Max tokens: {max_tokens}")
                        except ValueError:
                            print("Invalid value")
                    continue
                
                # Generate text
                generated = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                
                print("\nGenerated:")
                print(generated)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")