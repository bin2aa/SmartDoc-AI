#!/bin/bash

# SmartDocAI - Start All Services (Dev Mode)
# This script starts Ollama, FastAPI, and Streamlit UI

set -e

echo ">> SmartDocAI - Development Mode Startup"
echo "=========================================="
echo "Starting:"
echo "  1. FastAPI API Server (port 8000)"
echo "  2. Streamlit UI (port 8501)"
echo ""
echo "Make sure Ollama is running separately!"
echo ""

# Function to handle Ctrl+C
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill %1 %2 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    exit 1
fi

source venv/bin/activate

# Install dependencies
pip install -q -r requirements.txt

# Start FastAPI server
echo "[API] Starting FastAPI server (port 8000)..."
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 2

# Start Streamlit UI
echo "[UI] Starting Streamlit UI (port 8501)..."
streamlit run app.py --server.port 8501 &
STREAMLIT_PID=$!

echo ""
echo "=========================================="
echo "All services started!"
echo ""
echo "Service URLs:"
echo "  - FastAPI Docs: http://localhost:8000/api/docs"
echo "  - Streamlit UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="
echo ""

# Wait for both processes
wait $API_PID $STREAMLIT_PID
