@echo off
echo ============================================
echo  RACPad Incident Trend Analysis - Setup
echo ============================================
echo.

:: ============================================================
:: STEP 1 — Check Python is available
:: ============================================================

python --version >nul 2>&1
if not errorlevel 1 goto :python_found

py --version >nul 2>&1
if not errorlevel 1 (
    doskey python=py $*
    goto :python_found
)

echo.
echo ERROR: Python 3.10 or newer is required but was not found.
echo Please install Python from https://www.python.org/downloads/windows/
echo   - Check "Add python.exe to PATH" during installation.
echo Then re-run this setup.bat.
pause
exit /b 1

:python_found
echo Found:
python --version
echo.

:: ============================================================
:: STEP 2 — Create virtual environment
:: ============================================================
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists, skipping creation.
)

:: Activate venv
call venv\Scripts\activate.bat

:: ============================================================
:: STEP 3 — Upgrade pip
:: ============================================================
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

:: ============================================================
:: STEP 4 — Pre-install packages with known wheel issues on
::           corporate networks (Infosys JFrog proxy).
::           Must be done BEFORE requirements.txt.
::
::   sympy  — wheel has a man-page file that pip flags as
::             "outside site-packages" on Windows
::   httpx  — wheel has Scripts/httpx.exe with a path traversal
::             that pip rejects on Windows
:: ============================================================
echo.
echo Installing packages with corporate network workarounds...

pip install "sympy<1.14" --quiet
if errorlevel 1 (
    echo WARNING: Could not pre-install sympy ^(continuing anyway^)
)

pip install httpx --no-binary httpx --quiet
if errorlevel 1 (
    echo WARNING: Could not pre-install httpx ^(continuing anyway^)
)

:: ============================================================
:: STEP 5 — Install all remaining dependencies
:: ============================================================
echo.
echo Installing all dependencies from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Some dependencies failed to install.
    echo Common causes:
    echo   - No internet access
    echo   - Corporate proxy blocking PyPI
    echo Try running setup.bat again. If the problem persists,
    echo contact your IT team about PyPI access.
    pause
    exit /b 1
)

:: ============================================================
:: DONE
:: ============================================================
echo.
echo ============================================
echo  Setup complete! Run 'run.bat' to start.
echo ============================================
pause
