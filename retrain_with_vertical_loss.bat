@echo off
REM Retrain model with vertical progression loss
REM This will create a new checkpoint with proper vertical constraints

echo ============================================================
echo RETRAINING MODEL WITH VERTICAL PROGRESSION LOSS
echo ============================================================
echo.
echo This will train a NEW model with:
echo   - Vertical progression loss (weight: 0.5)
echo   - Start position constraint (weight: 0.3)
echo   - End position constraint (weight: 0.3)
echo.
echo Training will be saved to: checkpoints/climb_path_vertical
echo TensorBoard logs: runs/climb_path_vertical
echo.
echo Press Ctrl+C to cancel, or
pause

python backend/training/train_autoregressive.py ^
    --train_csv data/moonboard_train_quality.csv ^
    --val_csv data/moonboard_val_quality.csv ^
    --test_csv data/moonboard_test_quality.csv ^
    --batch_size 16 ^
    --num_epochs 50 ^
    --device cpu ^
    --num_workers 0 ^
    --log_dir runs/climb_path_vertical ^
    --checkpoint_dir checkpoints/climb_path_vertical

echo.
echo ============================================================
echo Training complete!
echo.
echo To monitor training:
echo   tensorboard --logdir runs/climb_path_vertical
echo.
echo To test the new model:
echo   python backend/training/generate_paths.py --checkpoint checkpoints/climb_path_vertical/best.pt --grade 7A --num_samples 5
echo ============================================================
pause
