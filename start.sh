#!/bin/bash
set -e

CONDA_ENV_NAME="pinns4preCastNodes"
BACKEND_PORT=8000
FRONTEND_PORT=3000

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo -e "\n🛑 Stopping services..."
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "🚀 Starting PINNs4PreCast (Conda Edition)..."

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME" || {
    echo "❌ Failed to activate conda env"
    exit 1
}

echo "✅ Conda environment activated"

# Check ports
if lsof -i :"$BACKEND_PORT" >/dev/null; then
    echo "❌ Port $BACKEND_PORT already in use"
    exit 1
fi

if lsof -i :"$FRONTEND_PORT" >/dev/null; then
    echo "❌ Port $FRONTEND_PORT already in use"
    exit 1
fi

# Start backend
echo "🐍 Launching Backend (FastAPI)..."
uvicorn src.api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

sleep 1
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi
echo "✅ Backend running on PID $BACKEND_PID"

# Start frontend
echo "⚛️ Launching Frontend..."
(
    cd frontend
    npm run dev -- -p "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

sleep 1
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "❌ Frontend failed to start"
    exit 1
fi
echo "✅ Frontend running on PID $FRONTEND_PID"

echo "------------------------------------------------"
echo "🎉 System is Live!"
echo "   Backend:  http://localhost:$BACKEND_PORT/docs"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "------------------------------------------------"

wait
