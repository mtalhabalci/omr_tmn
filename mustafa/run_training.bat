@echo off
echo Activating maskrcnn_local environment...
call conda activate maskrcnn_local

echo Running train_local.py...
python train_local.py

pause
