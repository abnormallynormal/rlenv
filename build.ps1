$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$Python = Join-Path $ProjectRoot ".arcade-build-venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    py -3.11 -m venv --system-site-packages .arcade-build-venv
}

& $Python -c "import torch; print('Bundling torch', torch.__version__)"
& $Python -m pip install --disable-pip-version-check -r requirements-build.txt
& $Python -m unittest discover -v
& $Python -m PyInstaller --noconfirm --clean RLArcade.spec

Write-Host ""
Write-Host "Built: $ProjectRoot\dist\RLArcade.exe" -ForegroundColor Green
