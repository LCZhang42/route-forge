@echo off
echo Training DistilGPT-2 on Modal.
echo No model weights will be downloaded to this machine.
echo.
python -m modal run modal_app.py::train_baseline --num-epochs 3
