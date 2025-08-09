"""Model configuration for GPT-style transformer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GPT model."""
    
    # Model architecture
    vocab_size: int = 16000
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    intermediate_size: int = 1536  # 4 * hidden_size
    max_position_embeddings: int = 512
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Activation
    activation: str = "gelu"
    
    # Layer norm
    layer_norm_eps: float = 1e-5
    
    # RoPE
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Data
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    validation_split: Optional[str] = None
    max_length: int = 512
    
    # Training
    batch_size: int = 32  # Per device
    gradient_accumulation_steps: int = 8  # Effective batch size: 32 * 8 = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler
    warmup_steps: int = 2000
    max_steps: int = 50000
    min_lr_ratio: float = 0.1
    
    # Regularization
    gradient_clip_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Mixed precision
    fp16: bool = True
    
    # Wandb
    use_wandb: bool = True
    project_name: str = "smalllm-gpt"
    run_name: Optional[str] = None
    
    # Output
    output_dir: str = "./checkpoints"
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001