# Northwind Mobile Assistant - Setup
# Installs Python deps + ensures Foundry Local is available.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== Northwind Mobile Assistant Setup ===" -ForegroundColor Cyan

# 1. Foundry Local
if (-not (Get-Command foundry -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Foundry Local..." -ForegroundColor Yellow
    winget install --silent --accept-source-agreements --accept-package-agreements Microsoft.FoundryLocal
} else {
    Write-Host "Foundry Local already installed." -ForegroundColor Green
}

# 2. venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# 3. Python deps
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. Run: .\run.bat" -ForegroundColor Green
