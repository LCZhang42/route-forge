@echo off
echo ========================================
echo Generate Climbing Paths with DistilGPT-2
echo (Endpoint Conditioning Mode)
echo ========================================
echo.

set /p GRADE="Enter grade (e.g., 7A) or press Enter for random: "
set /p START="Enter start hold (e.g., 0,4) or press Enter for random: "
set /p END="Enter end hold (e.g., 8,17) or press Enter for random: "

set CMD=python backend\training\generate_with_gpt2.py --model_path checkpoints\distilgpt2_climb\final --num_samples 5 --temperature 0.8

if not "%GRADE%"=="" set CMD=%CMD% --grade %GRADE%
if not "%START%"=="" set CMD=%CMD% --start_hold %START%
if not "%END%"=="" set CMD=%CMD% --end_hold %END%

echo.
echo Running: %CMD%
echo.
%CMD%

echo.
pause
