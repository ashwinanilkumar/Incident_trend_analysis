@echo off
echo ============================================
echo  RACPad Incident Trend Analysis
echo ============================================
echo.

:: Check venv exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found.
    echo Please run 'setup.bat' first.
    pause
    exit /b 1
)

:: Activate venv and launch app
call venv\Scripts\activate.bat
echo Starting app at http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py
