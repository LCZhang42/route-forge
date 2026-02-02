"""
Training script for autoregressive climb path generation model.

Trains the transformer model using teacher forcing on climb path sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from typing import Tuple

sys.path.append(str(Path(__file__).parent.parent))

from models.tokenizer import ClimbPathTokenizer
from models.climb_transformer import ClimbPathTransformerWithGeneration
from models.dataset import ClimbPathDataModule
from models.reachability_loss import ReachabilityLoss, SoftReachabilityLoss, AdaptiveReachabilityLoss


class Trainer:
    """Trainer class for climb path generation model."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: ClimbPathTokenizer,
        data_module: ClimbPathDataModule,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        log_dir: str = 'runs',
        checkpoint_dir: str = 'checkpoints',
        use_reachability_loss: bool = True,
        reachability_loss_weight: float = 0.3,
        reachability_loss_type: str = 'standard',
    ):
        """
        Initialize trainer.
        
        Args:
            model: ClimbPathTransformer model
            tokenizer: ClimbPathTokenizer instance
            data_module: ClimbPathDataModule instance
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Loss weights
        self.vertical_loss_weight = 0.5
        self.start_constraint_weight = 0.3
        self.end_constraint_weight = 0.3
        self.reachability_loss_weight = reachability_loss_weight
        
        # Reachability loss
        self.use_reachability_loss = use_reachability_loss
        if use_reachability_loss:
            if reachability_loss_type == 'soft':
                self.reachability_criterion = SoftReachabilityLoss(
                    tokenizer, max_reach=5.0, penalty_scale=0.5
                )
            elif reachability_loss_type == 'adaptive':
                self.reachability_criterion = AdaptiveReachabilityLoss(
                    tokenizer, initial_reach=6.0, final_reach=4.5, penalty_scale=0.5
                )
            else:  # standard
                self.reachability_criterion = ReachabilityLoss(
                    tokenizer, max_reach=5.0, penalty_scale=1.0
                )
            print(f"Using {reachability_loss_type} reachability loss (weight: {reachability_loss_weight})")
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.best_val_loss = float('inf')
    
    def get_lr(self) -> float:
        """Get learning rate with warmup schedule."""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        return self.base_lr
    
    def compute_vertical_progression_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute loss to encourage vertical progression in climbing paths.
        Penalizes flat paths and rewards Y-coordinate variety.
        
        Args:
            input_ids: Token sequence [batch_size, seq_len]
            
        Returns:
            Vertical progression loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Extract Y-coordinate tokens (odd positions after grade token)
        # Positions: 0=BOS, 1=grade, 2=x1, 3=y1, 4=x2, 5=y2, ...
        y_positions = list(range(3, seq_len, 2))  # Start from position 3 (first y-coord)
        
        if len(y_positions) < 2:
            return torch.tensor(0.0, device=input_ids.device)
        
        # Get Y tokens and convert to actual Y coordinates
        y_tokens = input_ids[:, y_positions]  # [batch_size, num_holds]
        
        # Filter out padding, EOS, and invalid tokens
        # Valid Y tokens are in range [27, 44] (Y coords 1-18)
        valid_mask = (y_tokens >= self.tokenizer.Y_COORD_START) & (y_tokens < self.tokenizer.EOS_TOKEN)
        
        total_loss = 0.0
        num_valid_sequences = 0
        
        for b in range(batch_size):
            valid_y = y_tokens[b][valid_mask[b]]
            
            if len(valid_y) < 2:
                continue
            
            # Convert tokens to actual Y coordinates (1-18)
            y_coords = valid_y - self.tokenizer.Y_COORD_START + 1
            
            # Compute vertical progression metrics
            y_diffs = y_coords[1:] - y_coords[:-1]
            
            # Loss 1: Penalize negative progression (going down too much)
            # Allow small downward moves but penalize large ones
            downward_penalty = torch.clamp(-y_diffs - 2, min=0).float().mean()
            
            # Loss 2: Encourage overall upward progression
            total_vertical_gain = y_coords[-1] - y_coords[0]
            # Penalize if total gain is less than 8 units (paths should climb up)
            min_expected_gain = 8.0
            upward_loss = torch.clamp(min_expected_gain - total_vertical_gain, min=0).float()
            
            # Loss 3: Penalize too many consecutive flat moves
            flat_moves = (torch.abs(y_diffs) < 1).float()
            flat_penalty = flat_moves.mean()
            
            # Combine losses
            sequence_loss = downward_penalty + 0.5 * upward_loss + 0.3 * flat_penalty
            total_loss += sequence_loss
            num_valid_sequences += 1
        
        if num_valid_sequences == 0:
            return torch.tensor(0.0, device=input_ids.device)
        
        return total_loss / num_valid_sequences
    
    def compute_position_constraint_loss(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss to encourage valid start and end positions.
        Start positions should be low (Y <= 6), end positions should be high (Y >= 12).
        
        Args:
            input_ids: Token sequence [batch_size, seq_len]
            
        Returns:
            (start_loss, end_loss)
        """
        batch_size, seq_len = input_ids.shape
        
        start_loss = 0.0
        end_loss = 0.0
        num_valid_sequences = 0
        
        for b in range(batch_size):
            # Find Y-coordinate tokens
            y_positions = list(range(3, seq_len, 2))
            if len(y_positions) < 1:
                continue
            
            y_tokens = input_ids[b, y_positions]
            valid_mask = (y_tokens >= self.tokenizer.Y_COORD_START) & (y_tokens < self.tokenizer.EOS_TOKEN)
            valid_y = y_tokens[valid_mask]
            
            if len(valid_y) < 1:
                continue
            
            # Convert to Y coordinates
            y_coords = valid_y - self.tokenizer.Y_COORD_START + 1
            
            # Start position constraint: first hold should have Y <= 6
            start_y = y_coords[0].float()
            max_start_y = 6.0
            start_penalty = torch.clamp(start_y - max_start_y, min=0)
            start_loss += start_penalty
            
            # End position constraint: last hold should have Y >= 12
            end_y = y_coords[-1].float()
            min_end_y = 12.0
            end_penalty = torch.clamp(min_end_y - end_y, min=0)
            end_loss += end_penalty
            
            num_valid_sequences += 1
        
        if num_valid_sequences == 0:
            return torch.tensor(0.0, device=input_ids.device), torch.tensor(0.0, device=input_ids.device)
        
        return start_loss / num_valid_sequences, end_loss / num_valid_sequences
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_vertical_loss = 0
        total_start_loss = 0
        total_end_loss = 0
        total_reachability_loss = 0
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(
                src=input_ids,
                src_key_padding_mask=(attention_mask == 0),
            )
            
            # Compute cross-entropy loss (predict next token)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Flatten for loss computation
            ce_loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            # Compute vertical progression loss
            vertical_loss = self.compute_vertical_progression_loss(input_ids)
            
            # Compute position constraint losses
            start_loss, end_loss = self.compute_position_constraint_loss(input_ids)
            
            # Compute reachability loss
            if self.use_reachability_loss:
                reachability_loss = self.reachability_criterion(input_ids)
            else:
                reachability_loss = torch.tensor(0.0, device=input_ids.device)
            
            # Combined loss
            loss = ce_loss + \
                   self.vertical_loss_weight * vertical_loss + \
                   self.start_constraint_weight * start_loss + \
                   self.end_constraint_weight * end_loss + \
                   self.reachability_loss_weight * reachability_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update step counter BEFORE calculating lr
            self.current_step += 1
            
            # Update weights with warmup
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_vertical_loss += vertical_loss.item()
            total_start_loss += start_loss.item()
            total_end_loss += end_loss.item()
            total_reachability_loss += reachability_loss.item()
            num_batches += 1
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', loss.item(), self.current_step)
            self.writer.add_scalar('Train/CE_Loss', ce_loss.item(), self.current_step)
            self.writer.add_scalar('Train/Vertical_Loss', vertical_loss.item(), self.current_step)
            self.writer.add_scalar('Train/Start_Loss', start_loss.item(), self.current_step)
            self.writer.add_scalar('Train/End_Loss', end_loss.item(), self.current_step)
            self.writer.add_scalar('Train/Reachability_Loss', reachability_loss.item(), self.current_step)
            self.writer.add_scalar('Train/LearningRate', lr, self.current_step)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'vert': f'{vertical_loss.item():.4f}',
                'reach': f'{reachability_loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_vertical_loss = total_vertical_loss / num_batches
        avg_start_loss = total_start_loss / num_batches
        avg_end_loss = total_end_loss / num_batches
        avg_reachability_loss = total_reachability_loss / num_batches
        
        print(f"  CE Loss: {avg_ce_loss:.4f} | Vertical Loss: {avg_vertical_loss:.4f} | "
              f"Start Loss: {avg_start_loss:.4f} | End Loss: {avg_end_loss:.4f} | "
              f"Reachability Loss: {avg_reachability_loss:.4f}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_vertical_loss = 0
        total_start_loss = 0
        total_end_loss = 0
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(
                src=input_ids,
                src_key_padding_mask=(attention_mask == 0),
            )
            
            # Compute cross-entropy loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            ce_loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            # Compute vertical progression loss
            vertical_loss = self.compute_vertical_progression_loss(input_ids)
            
            # Compute position constraint losses
            start_loss, end_loss = self.compute_position_constraint_loss(input_ids)
            
            # Combined loss
            loss = ce_loss + \
                   self.vertical_loss_weight * vertical_loss + \
                   self.start_constraint_weight * start_loss + \
                   self.end_constraint_weight * end_loss
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_vertical_loss += vertical_loss.item()
            total_start_loss += start_loss.item()
            total_end_loss += end_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'vert': f'{vertical_loss.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_vertical_loss = total_vertical_loss / num_batches
        avg_start_loss = total_start_loss / num_batches
        avg_end_loss = total_end_loss / num_batches
        
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/CE_Loss', avg_ce_loss, epoch)
        self.writer.add_scalar('Val/Vertical_Loss', avg_vertical_loss, epoch)
        self.writer.add_scalar('Val/Start_Loss', avg_start_loss, epoch)
        self.writer.add_scalar('Val/End_Loss', avg_end_loss, epoch)
        
        print(f"  CE Loss: {avg_ce_loss:.4f} | Vertical Loss: {avg_vertical_loss:.4f} | "
              f"Start Loss: {avg_start_loss:.4f} | End Loss: {avg_end_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'current_step': self.current_step,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f'Saved best model with val_loss={val_loss:.4f}')
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def train(self, num_epochs: int):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(epoch)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Early sanity checks after first epoch
            if epoch == 1:
                print("\n" + "="*60)
                print("SANITY CHECKS AFTER EPOCH 1")
                print("="*60)
                
                # Check for NaN/Inf
                if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
                    raise ValueError(f"Training loss is NaN/Inf after epoch 1: {train_loss}")
                if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
                    raise ValueError(f"Validation loss is NaN/Inf after epoch 1: {val_loss}")
                
                # Check if loss is decreasing (compare to initial random baseline)
                print(f"✓ Loss values are valid (train={train_loss:.4f}, val={val_loss:.4f})")
                
                # Check learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr == 0:
                    raise ValueError("Learning rate is 0! Model will not train.")
                print(f"✓ Learning rate is non-zero: {current_lr:.6f}")
                
                # Check gradients
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                
                if total_norm == 0:
                    raise ValueError("Gradient norm is 0! Model is not learning.")
                print(f"✓ Gradients are flowing (norm={total_norm:.4f})")
                
                print("="*60 + "\n")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train climb path generation model')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='data/moonboard_train.csv',
                        help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default='data/moonboard_val.csv',
                        help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default='data/moonboard_test.csv',
                        help='Path to test CSV')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length')
    
    # Loss arguments
    parser.add_argument('--use_reachability_loss', action='store_true', default=True,
                        help='Use reachability loss during training')
    parser.add_argument('--no_reachability_loss', action='store_true',
                        help='Disable reachability loss')
    parser.add_argument('--reachability_loss_weight', type=float, default=0.3,
                        help='Weight for reachability loss')
    parser.add_argument('--reachability_loss_type', type=str, default='standard',
                        choices=['standard', 'soft', 'adaptive'],
                        help='Type of reachability loss')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--log_dir', type=str, default='runs/climb_path',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/climb_path',
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = ClimbPathTokenizer()
    
    # Initialize data module
    print("Setting up data...")
    data_module = ClimbPathDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
    )
    data_module.setup()
    
    # Initialize model
    print("Initializing model...")
    model = ClimbPathTransformerWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    )
    
    # Initialize trainer
    use_reachability = args.use_reachability_loss and not args.no_reachability_loss
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_module=data_module,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_reachability_loss=use_reachability,
        reachability_loss_weight=args.reachability_loss_weight,
        reachability_loss_type=args.reachability_loss_type,
    )
    
    # Save config
    config = vars(args)
    config_path = Path(args.checkpoint_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    trainer.train(args.num_epochs)


if __name__ == '__main__':
    main()
