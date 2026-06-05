# 🚀 Quick Start Script for Satellite Error Prediction System
# Run this script in PowerShell to automatically set up everything!

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🛰️  SATELLITE ERROR PREDICTION SYSTEM" -ForegroundColor Cyan
Write-Host "    ISRO SIH 2025 - PS 25176" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found!" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation!" -ForegroundColor Red
    pause
    exit
}

Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\venv\Scripts\Activate.ps1"

# Check execution policy
try {
    & $activateScript
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "⚠ Execution policy error detected" -ForegroundColor Yellow
    Write-Host "Fixing execution policy..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    & $activateScript
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
}

Write-Host ""

# Install requirements
Write-Host "Installing required packages..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes. Please be patient..." -ForegroundColor Cyan
Write-Host ""

pip install --upgrade pip | Out-Null
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Error installing packages" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again" -ForegroundColor Red
    pause
    exit
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ask user what to do
Write-Host "What would you like to do?" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Launch Streamlit Web App (Recommended)" -ForegroundColor White
Write-Host "2. Train models using Python script" -ForegroundColor White
Write-Host "3. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🚀 Launching Streamlit App..." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop the app" -ForegroundColor Yellow
        Write-Host ""
        Start-Sleep -Seconds 2
        streamlit run app.py
    }
    "2" {
        Write-Host ""
        Write-Host "🤖 Starting model training..." -ForegroundColor Cyan
        Write-Host ""
        python satellite_predictor.py
        Write-Host ""
        Write-Host "✓ Training complete! Models saved to 'models/' folder" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run the Streamlit app:" -ForegroundColor Yellow
        Write-Host "streamlit run app.py" -ForegroundColor Cyan
        Write-Host ""
        pause
    }
    "3" {
        Write-Host ""
        Write-Host "Goodbye! To run the app later, use:" -ForegroundColor Yellow
        Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
        Write-Host "streamlit run app.py" -ForegroundColor Cyan
        Write-Host ""
        exit
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        Write-Host ""
        exit
    }
}
