@echo off
REM Visualize generated climb paths on MoonBoard background

python visualize_path.py --input checkpoints/climb_path_cpu/generated_paths.json --output visualizations
pause
