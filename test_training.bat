@echo off
REM Quick training test - runs only 3 epochs to verify setup

echo ========================================
echo QUICK TRAINING TEST (3 epochs)
echo ========================================
echo.

python backend/training/train_autoregressive.py ^
    --train_csv data/moonboard_train_quality.csv ^
    --val_csv data/moonboard_val_quality.csv ^
    --test_csv data/moonboard_test_quality.csv ^
    --device cpu ^
    --batch_size 16 ^
    --num_workers 0 ^
    --num_epochs 3 ^
    --d_model 256 ^
    --num_layers 6 ^
    --nhead 8 ^
    --dim_feedforward 1024 ^
    --learning_rate 1e-4 ^
    --warmup_steps 100 ^
    --max_seq_len 64 ^
    --checkpoint_dir checkpoints/test_run ^
    --log_dir runs/test_run

echo.
echo ========================================
echo Test complete! Check output above for:
echo   - Learning rate should be non-zero
echo   - Loss should be decreasing
echo   - No NaN/Inf errors
echo ========================================
pause
