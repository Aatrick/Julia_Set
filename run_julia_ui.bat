@echo off
echo Starting Julia Set Interactive UI using virtual environment...
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) ELSE (
    echo Virtual environment not found at .venv - trying to use system Python...
)
python julia_ui.py
pause
