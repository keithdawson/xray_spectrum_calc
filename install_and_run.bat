@echo off
SETLOCAL EnableDelayedExpansion

echo ==============================================
echo Installing X-Ray Spectrum App (Offline)
echo ==============================================

:: 1. Check for Python 3.11
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Installing Python 3.11 silently...
    start /wait python-3.11.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    :: Refresh PATH for current session
    call RefreshEnv.cmd >nul 2>&1
    :: Fallback if RefreshEnv is not available, we use the default install path
    set PATH=%PATH%;C:\Program Files\Python311\Scripts\;C:\Program Files\Python311\
)

echo Python is installed.
python --version

:: 2. Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: 3. Activate venv
call venv\Scripts\activate.bat

:: 4. Install dependencies offline
echo Installing dependencies from local wheels...
pip install --no-index --find-links=wheels -r requirements.txt

:: 5. Launch Streamlit
echo Launching Streamlit App...
streamlit run app.py

pause
