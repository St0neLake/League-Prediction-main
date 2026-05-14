@echo off

:: 1. Activate Python virtual environment
echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

:: --- Download Section ---
echo.
echo Downloading required file from Google Drive...

:: Set the output filename and the direct download URL
set "FILENAME=matches.csv"
set "URL=https://drive.google.com/uc?export=download&id=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"

:: Use cURL to download the file. The -L flag follows redirects.
curl -L -o "%FILENAME%" "%URL%"

echo Download complete. File saved as %FILENAME%
echo.
:: ----------------------

:: 2. Run Python scripts
echo Running Python scripts...
python best_kill_lightGBM.py
python train_ultimate.py
python best_lightGBM_winner.py
python best_time_lightGBM.py
python best_turret_lightGBM.py

echo.
echo All scripts have finished. ✅
pause