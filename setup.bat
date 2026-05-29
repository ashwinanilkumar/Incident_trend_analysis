@echo off
echo ============================================
echo  RACPad Incident Trend Analysis - Setup
echo ============================================
echo.

:: ============================================================
:: STEP 1 — Ensure Python 3.11 is available
:: ============================================================

:: Check 'python' command first
python --version >nul 2>&1
if not errorlevel 1 goto :python_found

:: Check 'py' launcher (Windows Python Launcher, often present even
:: when 'python' is not in PATH)
py --version >nul 2>&1
if not errorlevel 1 (
    :: Create a shim so the rest of the script can use 'python'
    doskey python=py $*
    goto :python_found
)

:: ── Python not found — download and install silently ────────
echo Python not found. Downloading Python 3.11 installer...
echo (This is a one-time download of ~25 MB)
echo.

set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=%TEMP%\python-3.11.9-amd64.exe

:: Use PowerShell to download (available on all Windows 7+ machines)
powershell -NoProfile -Command ^
  "try { Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"

if errorlevel 1 (
    echo.
    echo ERROR: Could not download Python automatically.
    echo Please install Python 3.11+ manually:
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Download Python 3.11.x for Windows
    echo   3. Run the installer - check "Add Python to PATH"
    echo   4. Re-run this setup.bat
    pause
    exit /b 1
)

echo Download complete. Installing Python for current user (no admin required)...
:: InstallAllUsers=0  -> installs for current user only (no admin needed)
:: PrependPath=1      -> adds Python to user PATH
:: Include_test=0     -> skips test suite to save space (~200 MB saved)
:: /quiet             -> no UI
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1

if errorlevel 1 (
    echo.
    echo ERROR: Python installation failed.
    echo Your IT policy may block installer execution.
    echo Please ask IT to install Python 3.11+ or install manually from:
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Add the default user-install path to PATH for this session
:: (registry PATH update won't be visible until a new shell opens)
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python was installed but cannot be found in PATH yet.
    echo Please close this window, open a new Command Prompt, and run setup.bat again.
    pause
    exit /b 1
)

echo Python installed successfully!
echo.

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
