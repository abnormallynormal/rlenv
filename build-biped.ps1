$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$Python = Join-Path $ProjectRoot ".biped-build-venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    py -3.11 -m venv --system-site-packages .biped-build-venv
}

& $Python -c "import torch, mujoco; print('Using torch', torch.__version__, 'and mujoco', mujoco.__version__)"
& $Python -m pip install --disable-pip-version-check -r requirements-build.txt
& $Python -m PyInstaller --noconfirm --clean BipedDemo.spec

Write-Host ""
Write-Host "Built: $ProjectRoot\dist\BipedDemo.exe" -ForegroundColor Green
