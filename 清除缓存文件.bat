@echo off
setlocal

REM Clear all files/folders under Output (keep Output itself)
set "OUTPUT_DIR=%~dp0Outputs"

if not exist "%OUTPUT_DIR%\" (
  echo [INFO] Output folder not found: "%OUTPUT_DIR%"
  exit /b 0
)

echo [INFO] Cleaning: "%OUTPUT_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$out = $env:OUTPUT_DIR; if (-not [string]::IsNullOrWhiteSpace($out) -and (Test-Path -LiteralPath $out)) { Get-ChildItem -LiteralPath $out -Force -ErrorAction SilentlyContinue | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue; if (Get-ChildItem -LiteralPath $out -Force -ErrorAction SilentlyContinue | Select-Object -First 1) { exit 1 } }; exit 0"

if errorlevel 1 (
  echo [WARN] Cleanup failed (some files may be in use)
  exit /b 1
)

echo [INFO] Done
exit /b 0
