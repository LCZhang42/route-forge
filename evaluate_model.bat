@echo off
REM Evaluate trained model on test data

echo ========================================
echo Evaluating Climb Path Model on Test Set
echo ========================================
echo.

REM Run evaluation (using global Python installation)
python backend\training\evaluate_model.py ^
    --checkpoint checkpoints/climb_path_cpu/best.pt ^
    --config checkpoints/climb_path_cpu/config.json ^
    --test_csv data/moonboard_test_quality.csv ^
    --train_csv data/moonboard_train_quality.csv ^
    --val_csv data/moonboard_val_quality.csv ^
    --device cpu ^
    --batch_size 32 ^
    --num_workers 0

echo.
echo ========================================
echo Evaluation Complete!
echo ========================================
pause
