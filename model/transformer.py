"""GPT-style transformer model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from torch.utils.checkpoint import checkpoint

from ..configs.model_config import ModelConfig
from .attention import MultiHeadAttention
from .layers import MLP, LayerNorm


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            max_seq_len=config.max_position_embeddings,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
        )
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.dropout = config.dropout
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.ln_1(x)
        
        if use_cache:
            attn_output, present_key_value = self.attn(
                x, attention_mask=attention_mask, use_cache=use_cache, past_key_value=past_key_value
            )
        else:
            attn_output = self.attn(x, attention_mask=attention_mask)
            present_key_value = None
        
        x = residual + attn_output
        
        # Pre-norm MLP
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x
        
        if use_cache:
            return x, present_key_value
        else:
            return x


class GPTModel(nn.Module):
    """GPT-style transformer model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head (tied with input embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,}")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        x = self.wte(input_ids)
        x = self.dropout(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        
        # Forward through transformer blocks
        past_key_values_output = () if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing during training
                if use_cache:
                    # Can't use checkpointing with cache
                    block_output = block(x, attention_mask, use_cache, past_key_value)
                    x, present_key_value = block_output
                    past_key_values_output += (present_key_value,)
                else:
                    x = checkpoint(block, x, attention_mask, use_cache, past_key_value)
            else:
                if use_cache:
                    block_output = block(x, attention_mask, use_cache, past_key_value)
                    x, present_key_value = block_output
                    past_key_values_output += (present_key_value,)
                else:
                    x = block(x, attention_mask, use_cache, past_key_value)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'past_key_values': past_key_values_output,
            }
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            if past_key_values_output is not None:
                outputs = outputs + (past_key_values_output,)
            return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # Initialize past key values for efficient generation
            past_key_values = None
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                
                logits = outputs['logits']
                past_key_values = outputs['past_key_values']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == eos_token_id:
                    break
        
        return input_ids