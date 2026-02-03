@echo off
echo ========================================
echo Cleaning up obsolete files
echo ========================================
echo.

echo Removing obsolete documentation...
del /F /Q "CPU_TRAINING_GUIDE.md" 2>nul
del /F /Q "ENDPOINT_CONDITIONING_GUIDE.md" 2>nul
del /F /Q "HUGGINGFACE_FINETUNING.md" 2>nul
del /F /Q "IMPLEMENTATION_SUMMARY.md" 2>nul
del /F /Q "LOSS_FUNCTIONS.md" 2>nul
del /F /Q "PROJECT_STATUS.md" 2>nul
del /F /Q "TRAINING_SPEED_GUIDE.md" 2>nul
del /F /Q "TRAINING_SUMMARY.md" 2>nul
del /F /Q "VERTICAL_PROGRESSION_TRAINING.md" 2>nul

echo Removing obsolete batch scripts...
del /F /Q "check_memorization.bat" 2>nul
del /F /Q "evaluate_model.bat" 2>nul
del /F /Q "test_training.bat" 2>nul
del /F /Q "train_gpt2.bat" 2>nul
del /F /Q "train_with_constraints.bat" 2>nul
del /F /Q "train_with_constraints_fast.bat" 2>nul
del /F /Q "train_with_endpoints.bat" 2>nul
del /F /Q "visualize_training.bat" 2>nul
del /F /Q "train_distilgpt2_fast.bat" 2>nul

echo Removing obsolete test scripts...
del /F /Q "test_endpoint_conditioning.py" 2>nul
del /F /Q "test_endpoint_format.py" 2>nul
del /F /Q "test_endpoint_generation.py" 2>nul
del /F /Q "test_evaluation_losses.py" 2>nul
del /F /Q "test_model.py" 2>nul
del /F /Q "visualize_training.py" 2>nul
del /F /Q "compare_benchmark_paths.py" 2>nul
del /F /Q "compare_benchmark_reordered.py" 2>nul

echo Removing obsolete datasets...
del /F /Q "data_reordered\moonboard_cleaned.csv" 2>nul
del /F /Q "data_reordered\moonboard_test.csv" 2>nul
del /F /Q "data_reordered\moonboard_train.csv" 2>nul
del /F /Q "data_reordered\moonboard_val.csv" 2>nul
del /F /Q "data_reordered\moonboard_test_benchmark.csv" 2>nul
del /F /Q "data_reordered\moonboard_train_benchmark.csv" 2>nul
del /F /Q "data_reordered\moonboard_val_benchmark.csv" 2>nul

echo Removing obsolete training scripts...
del /F /Q "backend\training\finetune_with_constraints.py" 2>nul

echo.
echo ========================================
echo Cleanup complete!
echo ========================================
echo.
echo KEPT (Essential files):
echo - README.md, QUICKSTART.md, INSTALL.md, WINDOWS_SETUP.md
echo - ENDPOINT_CONDITIONING.md, GRADE_BALANCE_GUIDE.md, QUALITY_FILTERING_GUIDE.md
echo - check_data_quality.py
echo - reorder_all_datasets.py
echo - analyze_grade_distribution.py
echo - train_distilgpt2.bat
echo - generate_paths.bat
echo - start_server.bat, start_frontend.bat, start_web_interface.bat
echo - web_interface.py, visualize_path.py
echo - backend/training/finetune_huggingface.py
echo - backend/training/generate_with_gpt2.py
echo - data_reordered/moonboard_train_quality_filtered.csv
echo - data_reordered/moonboard_test_quality.csv
echo - data_reordered/moonboard_val_quality.csv (KEPT - needed for validation!)
echo - data_reordered/moonboard_train_quality.csv
echo.
pause
