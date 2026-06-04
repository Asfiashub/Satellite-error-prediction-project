#!/bin/bash

# 🚀 Quick Start Script for Satellite Error Prediction System
# Run this script in Terminal to automatically set up everything!

echo "========================================"
echo "🛰️  SATELLITE ERROR PREDICTION SYSTEM"
echo "    ISRO SIH 2025 - PS 25176"
echo "========================================"
echo ""

# Check if Python is installed
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python not found!"
    echo "Please install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
fi

echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""

# Install requirements
echo "Installing required packages..."
echo "This may take 5-10 minutes. Please be patient..."
echo ""

pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All packages installed successfully!"
else
    echo "✗ Error installing packages"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ SETUP COMPLETE!"
echo "========================================"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo ""
echo "1. Launch Streamlit Web App (Recommended)"
echo "2. Train models using Python script"
echo "3. Exit"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Streamlit App..."
        echo ""
        echo "The app will open in your browser at http://localhost:8501"
        echo "Press Ctrl+C to stop the app"
        echo ""
        sleep 2
        streamlit run app.py
        ;;
    2)
        echo ""
        echo "🤖 Starting model training..."
        echo ""
        python3 satellite_predictor.py
        echo ""
        echo "✓ Training complete! Models saved to 'models/' folder"
        echo ""
        echo "You can now run the Streamlit app:"
        echo "streamlit run app.py"
        echo ""
        read -p "Press Enter to continue..."
        ;;
    3)
        echo ""
        echo "Goodbye! To run the app later, use:"
        echo "source venv/bin/activate"
        echo "streamlit run app.py"
        echo ""
        exit 0
        ;;
    *)
        echo ""
        echo "Invalid choice. Exiting..."
        echo ""
        exit 1
        ;;
esac
