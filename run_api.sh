#!/bin/bash

# SmartDocAI - Start API Server
# This script starts the FastAPI server for N8N integration

set -e

echo "🚀 SmartDocAI API Server - Startup Script"
echo "=========================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Warning: Ollama not detected at localhost:11434"
    echo "   Make sure Ollama is running before using LLM features"
fi

# Start FastAPI server
echo "🌐 Starting FastAPI server on http://0.0.0.0:8001"
echo "📚 API Documentation: http://localhost:8001/api/docs"
echo "=========================================="
echo ""

python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001 --reload

# For production use:
# gunicorn src.api.server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
