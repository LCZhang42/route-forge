#!/bin/bash
# CPU-optimized training configuration for climb path generation

python backend/training/train_autoregressive.py \
    --device cpu \
    --batch_size 16 \
    --num_workers 0 \
    --num_epochs 50 \
    --d_model 256 \
    --num_layers 6 \
    --nhead 8 \
    --dim_feedforward 1024 \
    --learning_rate 1e-4 \
    --warmup_steps 500 \
    --max_seq_len 64 \
    --checkpoint_dir checkpoints/climb_path_cpu \
    --log_dir runs/climb_path_cpu

# CPU-specific optimizations:
# - batch_size 16 (smaller batches for CPU)
# - num_workers 0 (avoid multiprocessing overhead)
# - max_seq_len 64 (shorter sequences process faster)
# - warmup_steps 500 (fewer warmup steps)
