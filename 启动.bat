@echo off
setlocal
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Virtual env not found: %PYTHON_EXE%
    echo Please create .venv in project root and install requirements.
    pause
    exit /b 1
)

if not exist "%ROOT_DIR%api\api.py" (
    echo Missing api\api.py. Unable to start service.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m api.api
pause
