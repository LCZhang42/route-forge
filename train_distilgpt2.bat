@echo off
echo ========================================
echo Fine-tuning DistilGPT-2 for Climb Path Generation
echo (Smaller, faster model)
echo ========================================
echo.

python backend\training\finetune_huggingface.py ^
    --train_csv data_reordered\moonboard_train_quality_filtered.csv ^
    --val_csv data_reordered\moonboard_test_quality.csv ^
    --model_name distilgpt2 ^
    --output_dir checkpoints\distilgpt2_climb ^
    --num_epochs 10 ^
    --batch_size 16 ^
    --eval_batch_size 32 ^
    --learning_rate 5e-5 ^
    --warmup_steps 500 ^
    --weight_decay 0.01 ^
    --max_length 256 ^
    --save_steps 500 ^
    --eval_steps 500 ^
    --logging_steps 100

echo.
echo ========================================
echo Training complete!
echo ========================================
pause
