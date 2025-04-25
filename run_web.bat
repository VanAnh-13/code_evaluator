@echo off
echo Starting C++ Code Analyzer Web Application...
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
python -c "import flask" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Flask not found. Installing dependencies...
    python install_dependencies.py
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

REM Run the web application
echo Starting web server...
python run_web.py
pause