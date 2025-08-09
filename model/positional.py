"""Rotary Positional Encoding implementation."""

import math
import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) implementation.
    
    Based on the paper: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute frequency matrix
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for rotary embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin values for given sequence length."""
        if seq_len != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seq_len
            
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Compute frequencies for each position
            freqs = torch.outer(t, self.inv_freq)
            
            # Duplicate frequencies for each pair of dimensions
            freqs = torch.cat([freqs, freqs], dim=-1)
            
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()
        
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key tensors."""
        seq_len = q.shape[-2]
        cos, sin = self._compute_cos_sin(seq_len, q.device)
        
        # Apply rotary embedding
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass applying RoPE to query and key."""
        return self.apply_rotary_pos_emb(q, k)