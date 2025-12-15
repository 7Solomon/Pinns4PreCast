#!/bin/bash

CONDA_ENV_NAME="pinns4preCastNodes"
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Function to kill processes on exit
cleanup() {
    echo -e "\n Stopping services..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Trap Ctrl+C
trap cleanup SIGINT

echo "🚀 Starting PINNs4PreCast (Conda Edition)..."

#  ACTIVATE CONDA
# This MAGICE line allows 'conda activate' to work inside a shell script
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

if [ $? -eq 0 ]; then
    echo "✅ Activated Conda environment: $CONDA_ENV_NAME"
else
    echo "⚠️  Could not activate Conda environment '$CONDA_ENV_NAME'."
    echo "   Running with current system Python (this might fail)."
fi

#  START BACKEND
echo "🐍 Launching Backend (FastAPI)..."
uvicorn src.api.main:app --reload --port $BACKEND_PORT &
BACKEND_PID=$!
echo "✅ Backend running on PID $BACKEND_PID"

# START FRONTEND
echo "⚛️  Launching Frontend..."
cd frontend
npm run dev -- -p $FRONTEND_PORT &
FRONTEND_PID=$!
echo "✅ Frontend running on PID $FRONTEND_PID"

#  KEEP ALIVE
echo "------------------------------------------------"
echo "🎉 System is Live!"
echo "   Backend: http://localhost:$BACKEND_PORT/docs"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "------------------------------------------------"

wait
