@echo off
REM Serve the FastAPI backend from Modal; model weights remain remote.
python -m modal serve modal_app.py
