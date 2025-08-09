"""Optimizer and scheduler utilities."""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Optional


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    warmup_steps: int = 2000,
    max_steps: int = 50000,
    min_lr_ratio: float = 0.1,
) -> Tuple[AdamW, LambdaLR]:
    """Create AdamW optimizer and cosine scheduler with warmup."""
    
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to bias terms and layer norms
            if 'bias' in name or 'ln' in name or 'layernorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
        }
    ]
    
    # Create optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )
    
    # Create cosine scheduler with warmup
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print(f"Created optimizer with {len(decay_params)} decay and {len(no_decay_params)} no-decay parameters")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"Warmup steps: {warmup_steps}, Max steps: {max_steps}")
    
    return optimizer, scheduler


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)