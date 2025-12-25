#!/bin/bash

# VeroAllarme - Local Development Runner
# Run backend without Docker for quick testing

echo "🚀 Starting VeroAllarme Backend (Local Mode)"
echo "============================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q fastapi uvicorn pydantic pydantic-settings python-dotenv

# Create minimal .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Run the backend
echo ""
echo "✓ Starting FastAPI server..."
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""

cd backend && python main.py
