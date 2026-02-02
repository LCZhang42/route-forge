@echo off
REM CPU-optimized training configuration for climb path generation (Windows)

python backend/training/train_autoregressive.py ^
    --train_csv data/moonboard_train_quality.csv ^
    --val_csv data/moonboard_val_quality.csv ^
    --test_csv data/moonboard_test_quality.csv ^
    --device cpu ^
    --batch_size 16 ^
    --num_workers 0 ^
    --num_epochs 50 ^
    --d_model 256 ^
    --num_layers 6 ^
    --nhead 8 ^
    --dim_feedforward 1024 ^
    --learning_rate 1e-4 ^
    --warmup_steps 500 ^
    --max_seq_len 64 ^
    --checkpoint_dir checkpoints/climb_path_cpu ^
    --log_dir runs/climb_path_cpu

REM CPU-specific optimizations:
REM - batch_size 16 (smaller batches for CPU)
REM - num_workers 0 (avoid multiprocessing overhead)
REM - max_seq_len 64 (shorter sequences process faster)
REM - warmup_steps 500 (fewer warmup steps)
