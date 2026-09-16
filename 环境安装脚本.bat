@echo off
setlocal
set "SEEDVC_INSTALL_SCRIPT=%~f0"
set "SEEDVC_ROOT=%~dp0"
set "SEEDVC_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "SEEDVC_POWERSHELL=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
"%SEEDVC_POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -Command "$source = [IO.File]::ReadAllText($env:SEEDVC_INSTALL_SCRIPT, [Text.Encoding]::UTF8); & ([scriptblock]::Create(($source -split '(?m)^# POWERSHELL\r?$', 2)[1]))"
set "SEEDVC_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %SEEDVC_EXIT_CODE%

# POWERSHELL
# BAT 只负责启动 64 位 PowerShell；以下逻辑保持在当前用户下执行。
$ErrorActionPreference = 'Stop'
$PSDefaultParameterValues['Invoke-WebRequest:UseBasicParsing'] = $true
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
Set-Location -LiteralPath $env:SEEDVC_ROOT

function Get-VcFileVersion([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $info = (Get-Item -LiteralPath $Path).VersionInfo
    return [version]::new($info.FileMajorPart, $info.FileMinorPart, $info.FileBuildPart)
}

function Get-VcInstalledVersion {
    $runtime = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    if ($runtime -and $runtime.Installed -eq 1) {
        $version = [version]$runtime.Version.TrimStart('v', 'V')
        return [version]::new($version.Major, $version.Minor, $version.Build)
    }
    return $null
}

function Test-VcFiles([version]$MinimumVersion) {
    $healthy = $true
    foreach ($name in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
        $path = Join-Path ([Environment]::SystemDirectory) $name
        $version = Get-VcFileVersion $path
        Write-Host ("  {0}: {1}" -f $path, $(if ($version) { $version } else { '缺失' }))
        if (-not $version -or $version -lt $MinimumVersion) { $healthy = $false }
    }
    return $healthy
}

function Install-VcRuntime {
    $workDir = Join-Path ([IO.Path]::GetTempPath()) ('SeedVoiceConversion-VC-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $workDir | Out-Null
    $installer = Join-Path $workDir 'vc_redist.x64.exe'
    $logPath = Join-Path $workDir 'vc-redist.log'
    Write-Host "正在下载 Microsoft Visual C++ Redistributable x64..."
    Write-Host "安装包：$installer"
    Write-Host "安装器日志（执行安装时生成）：$logPath"
    Invoke-WebRequest -Uri 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile $installer -UseBasicParsing -TimeoutSec 120
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(^|,\s*)CN=Microsoft Corporation(,|$)') {
        throw '运行库安装包的微软数字签名验证失败，已停止安装。'
    }
    $packageVersion = Get-VcFileVersion $installer
    if (-not $packageVersion -or $packageVersion.Major -ne 14) {
        throw '无法识别下载的 VC++ v14 运行库安装包版本。'
    }
    $installedVersion = Get-VcInstalledVersion
    Write-Host "下载版本：$packageVersion；已登记版本：$installedVersion"
    if ($installedVersion -and $installedVersion -ge $packageVersion) {
        if (Test-VcFiles $installedVersion) {
            Write-Host 'VC++ x64 运行库及实际 DLL 已满足要求，跳过安装。'
            return $false
        }
        if ($installedVersion -gt $packageVersion) {
            throw '已登记更高版本，但实际 DLL 缺失或落后。请在 Windows 已安装的应用中修复该版本的 VC++ x64 运行库并重启。'
        }
    }
    $action = if ($installedVersion -eq $packageVersion) { '/repair' } else { '/install' }
    Write-Host "正在执行运行库 $action，请允许安装器的管理员授权提示。"
    $process = Start-Process -FilePath $installer -ArgumentList @($action, '/passive', '/norestart', '/log', ('"{0}"' -f $logPath)) -Verb RunAs -Wait -PassThru
    $code = $process.ExitCode
    Write-Host "运行库安装器退出码：$code；日志：$logPath"
    if ($code -eq 3010) {
        Write-Host '运行库安装成功，需要重启；继续安装 Python 依赖。'
        return $true
    }
    # 部分安装器返回原始 MSI 错误码，部分返回 HRESULT。
    if ($code -ne 0 -and $code -notin @(1638, -2147023258, 2147944038)) {
        throw "VC++ x64 运行库安装失败（退出码 $code），请查看安装日志。"
    }
    $installedVersion = Get-VcInstalledVersion
    if (-not $installedVersion -or $installedVersion -lt $packageVersion -or -not (Test-VcFiles $installedVersion)) {
        throw '运行库安装登记或实际 DLL 仍不符合要求，请修复已安装的 VC++ x64 运行库并重启后重试。'
    }
    Write-Host 'VC++ x64 运行库文件检查通过。'
    return $false
}

function Install-PythonDependencies {
    Write-Host '正在安装 uv...'
    & $env:SEEDVC_POWERSHELL -NoProfile -ExecutionPolicy Bypass -Command '$ErrorActionPreference = ''Stop''; irm https://astral.sh/uv/install.ps1 | iex' | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "uv 安装失败（退出码 $LASTEXITCODE）。" }
    # 新安装的 uv 不一定已出现在当前进程的 PATH 中。
    $uvExecutable = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uvExecutable -PathType Leaf)) {
        $uvExecutable = (Get-Command uv -ErrorAction Stop).Source
    }
    & $uvExecutable venv --python=3.10 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Python 虚拟环境创建失败（退出码 $LASTEXITCODE）。" }
    if (-not (Test-Path -LiteralPath 'pyproject.toml')) {
        & $uvExecutable init --python=3.10 | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Python 项目初始化失败（退出码 $LASTEXITCODE）。" }
    }
    & $uvExecutable pip install -r requirements-win.txt --torch-backend cu128 --prerelease allow | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Python 依赖安装失败（退出码 $LASTEXITCODE）。" }
}

function Install-Environment {
    $needsRestart = $false
    try {
        $needsRestart = Install-VcRuntime
        Install-PythonDependencies
        if ($needsRestart) {
            Write-Host '安装步骤已完成，请手动重启 Windows 后再启动服务。' -ForegroundColor Yellow
            return 3010
        }
        Write-Host '正在验证 PyTorch 导入...'
        $pythonExecutable = Join-Path $env:SEEDVC_ROOT '.venv\Scripts\python.exe'
        & $pythonExecutable -c "import torch, torchvision, torchaudio; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'PyTorch 导入验证失败，环境尚不可用。请检查上方错误。' }
        Write-Host '环境安装成功，PyTorch 导入验证通过。' -ForegroundColor Green
        return 0
    }
    catch {
        Write-Host "环境安装失败：$($_.Exception.Message)" -ForegroundColor Red
        if ($needsRestart) {
            Write-Host 'VC++ 运行库仍需要重启。请手动重启 Windows 后重新运行本脚本，完成环境安装。' -ForegroundColor Yellow
        }
        return 1
    }
}

exit (Install-Environment)
