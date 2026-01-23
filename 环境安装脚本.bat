@echo off
setlocal
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv venv --python=3.10
uv init --python=3.10
uv pip install -r requirements-win.txt --torch-backend cu128 --prerelease allow

pause
