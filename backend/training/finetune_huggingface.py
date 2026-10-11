"""
Fine-tuning script for climb path generation using HuggingFace GPT-2.

Uses a pre-trained GPT-2 small model and fine-tunes it on climbing path sequences.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import ast
import json
import sys
from tqdm import tqdm
from typing import List

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
)
from datasets import Dataset


class ClimbPathDataset:
    """Dataset for converting climbing paths to endpoint conditioning format for GPT-2."""
    
    def __init__(
        self,
        csv_path: str,
        max_samples: int | None = None,
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with climbing data
            max_samples: Optional deterministic row limit for smoke tests
        """
        self.df = pd.read_csv(csv_path)
        if max_samples is not None:
            self.df = self.df.head(max_samples).copy()
        print(f"Loaded {len(self.df)} climbing paths from {csv_path}")
    
    def path_to_text_with_endpoints(self, grade: str, holds: List[List[int]]) -> str:
        """
        Convert a climbing path to endpoint conditioning format.
        
        Format: "GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)"
        
        Args:
            grade: Grade string
            holds: List of [x, y] coordinates (full path)
            
        Returns:
            Text representation with endpoint conditioning
        """
        if len(holds) < 2:
            return None
        
        start_hold = holds[0]
        end_hold = holds[-1]
        mid_holds = holds[1:-1]  # Intermediate holds (what model should predict)
        
        start_str = f"({start_hold[0]},{start_hold[1]})"
        end_str = f"({end_hold[0]},{end_hold[1]})"
        
        if len(mid_holds) > 0:
            mid_str = " ".join([f"({x},{y})" for x, y in mid_holds])
        else:
            mid_str = ""  # Direct path from start to end
        
        return f"GRADE: {grade} | START: {start_str} | END: {end_str} | MID: {mid_str}"
    
    def prepare_dataset(self) -> List[str]:
        """
        Prepare all climbing paths as text with endpoint conditioning.
        
        Returns:
            List of text representations
        """
        texts = []
        skipped = 0
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Converting paths"):
            try:
                grade = row['grade']
                full_path = ast.literal_eval(row['full_path'])
                
                if len(full_path) < 2:
                    skipped += 1
                    continue
                
                text = self.path_to_text_with_endpoints(grade, full_path)
                if text:
                    texts.append(text)
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing row: {e}")
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"Skipped {skipped} paths (too short or invalid)")
        
        return texts


class SafetyCallback(TrainerCallback):
    """Callback to check for NaN/Inf in losses and gradients."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check logs for NaN/Inf values."""
        if logs:
            loss = logs.get('loss', None)
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    print(f"\n⚠️  WARNING: Loss is {loss}! Stopping training.")
                    control.should_training_stop = True
                    return control
        return control


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 for climb path generation')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, 
                        default='data_reordered/moonboard_train_quality_filtered.csv',
                        help='Path to training CSV (filtered for grade balance)')
    parser.add_argument('--val_csv', type=str, 
                        default='data_reordered/moonboard_val_quality.csv',
                        help='Path to validation CSV')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='distilbert/distilgpt2',
                        help='Hugging Face base model ID')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints (use a Modal Volume)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optional row limit for a smoke run')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Optional validation row limit for a smoke run')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Checkpoint directory to resume from')
    
    # System arguments
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training (requires CUDA)')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Initialize tokenizers
    print("Initializing tokenizers...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    
    # Add padding token if not present
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
    # Load and prepare datasets
    print("\nPreparing training data...")
    train_dataset = ClimbPathDataset(args.train_csv, args.max_train_samples)
    train_texts = train_dataset.prepare_dataset()
    
    print("\nPreparing validation data...")
    val_dataset = ClimbPathDataset(args.val_csv, args.max_val_samples)
    val_texts = val_dataset.prepare_dataset()
    
    # Tokenize texts
    print("\nTokenizing texts...")
    train_encodings = gpt2_tokenizer(
        train_texts,
        truncation=True,
        max_length=args.max_length,
    )
    
    val_encodings = gpt2_tokenizer(
        val_texts,
        truncation=True,
        max_length=args.max_length,
    )
    
    # Create HuggingFace datasets
    train_hf_dataset = Dataset.from_dict(train_encodings)
    
    val_hf_dataset = Dataset.from_dict(val_encodings)
    
    print(f"\nTrain dataset size: {len(train_hf_dataset)}")
    print(f"Val dataset size: {len(val_hf_dataset)}")
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model: {args.model_name}")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(gpt2_tokenizer))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using endpoint conditioning format: GRADE + START + END → INTERMEDIATE HOLDS")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=['tensorboard'],
        save_total_limit=3,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=gpt2_tokenizer,
        mlm=False,  # GPT-2 uses causal LM, not masked LM
    )
    
    # Initialize trainer with safety callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=val_hf_dataset,
        data_collator=data_collator,
        callbacks=[SafetyCallback()],
    )
    
    # Save config
    config = vars(args)
    config_path = Path(args.output_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f'{args.output_dir}/final')
    gpt2_tokenizer.save_pretrained(f'{args.output_dir}/final')
    
    # Evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
    
    # Save evaluation results
    with open(f'{args.output_dir}/eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}/final")
    print("="*60)


if __name__ == '__main__':
    main()
