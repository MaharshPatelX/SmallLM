"""Training class for GPT model."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path
import json

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from ..configs.model_config import TrainingConfig
from .utils import AverageMeter, EarlyStopping, format_metrics, save_checkpoint
from .optimizer import create_optimizer_and_scheduler


class Trainer:
    """Trainer class for GPT model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        device: str = 'cuda',
        tokenizer_path: str = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device
        self.tokenizer_path = tokenizer_path
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            eps=self.config.eps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
        )
        
        # Mixed precision training
        self.use_mixed_precision = self.config.fp16 and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (FP16)")
        
        # Initialize wandb if requested
        self.use_wandb = self.config.use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__,
            )
            wandb.watch(self.model)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_threshold,
        )
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        
        print(f"Trainer initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_loss_meter.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    
                    # Update weights
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Update metrics
                self.train_loss_meter.update(loss.item() * self.config.gradient_accumulation_steps)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    metrics = {
                        'train_loss': self.train_loss_meter.avg,
                        'learning_rate': lr,
                        'step': self.global_step
                    }
                    
                    print(format_metrics(metrics, self.global_step))
                    
                    if self.use_wandb:
                        wandb.log(metrics, step=self.global_step)
                
                # Validation
                if self.global_step % self.config.eval_steps == 0 and self.val_loader is not None:
                    val_metrics = self.validate()
                    
                    # Check for improvement
                    if val_metrics['val_loss'] < self.best_loss:
                        self.best_loss = val_metrics['val_loss']
                        self.save_checkpoint(is_best=True)
                    
                    # Early stopping
                    if self.early_stopping(val_metrics['val_loss']):
                        print(f"Early stopping triggered at step {self.global_step}")
                        return {'early_stop': True}
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if max steps reached
                if self.global_step >= self.config.max_steps:
                    print(f"Max steps ({self.config.max_steps}) reached")
                    return {'max_steps_reached': True}
        
        epoch_time = time.time() - epoch_start_time
        return {
            'train_loss': self.train_loss_meter.avg,
            'epoch_time': epoch_time,
            'steps_per_second': len(self.train_loader) / epoch_time
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.val_loss_meter.reset()
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                self.val_loss_meter.update(loss.item())
        
        metrics = {'val_loss': self.val_loss_meter.avg}
        
        print(f"Validation - Loss: {metrics['val_loss']:.4f}")
        
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        self.model.train()
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        suffix = '_best' if is_best else f'_step_{self.global_step}'
        checkpoint_path = self.output_dir / f'checkpoint{suffix}.pt'
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            loss=self.train_loss_meter.avg,
            filepath=str(checkpoint_path),
            config=self.config,
            tokenizer_path=self.tokenizer_path,
        )
        
        # Save config as JSON
        config_path = self.output_dir / f'config{suffix}.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Training for max {self.config.max_steps} steps")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1
            print(f"\n=== Epoch {epoch} ===")
            
            epoch_metrics = self.train_epoch()
            
            # Check for early termination
            if epoch_metrics.get('early_stop') or epoch_metrics.get('max_steps_reached'):
                break
            
            print(f"Epoch {epoch} completed - Train Loss: {epoch_metrics['train_loss']:.4f}")
        
        # Final validation and save
        if self.val_loader is not None:
            final_metrics = self.validate()
            print(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        
        self.save_checkpoint(is_best=False)
        print("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"Resuming training from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint.get('loss', float('inf'))
        
        print(f"Resumed from step {self.global_step} with loss {self.best_loss:.4f}")