@echo off
REM Check if model is memorizing training data

echo ========================================
echo Memorization Check for Climb Path Model
echo ========================================
echo.

REM Run memorization check
python backend\training\check_memorization.py ^
    --checkpoint checkpoints/climb_path_cpu/best.pt ^
    --config checkpoints/climb_path_cpu/config.json ^
    --train_csv data/moonboard_train_quality.csv ^
    --val_csv data/moonboard_val_quality.csv ^
    --test_csv data/moonboard_test_quality.csv ^
    --grade 7A ^
    --num_samples 100 ^
    --temperature 1.0 ^
    --device cpu

echo.
echo ========================================
echo Check Complete!
echo ========================================
pause
