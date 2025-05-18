@echo off
echo Starting Julia Set Interactive UI using virtual environment...
call .venv\Scripts\activate

python julia_ui.py
pause
