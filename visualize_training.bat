@echo off
REM Visualize training performance from TensorBoard logs

python visualize_training.py ^
    --log_dir runs/climb_path_cpu ^
    --checkpoint_dir checkpoints/climb_path_cpu ^
    --output_dir training_plots

echo.
echo Visualization complete! Check the training_plots folder for results.
pause
