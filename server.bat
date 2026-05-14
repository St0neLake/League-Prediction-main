@echo off
echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo Starting Flask Server...
python app.py

pause
