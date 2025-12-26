#!/bin/bash

# --- Configuration ---
CONDA_ENV_NAME="pinns4preCastNodes"
BACKEND_PORT=8000
FRONTEND_PORT=3000

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Cleanup Trap ---
cleanup() {
    echo -e "\n${RED}Stopping services...${NC}"
    if [ -n "$BACKEND_PID" ]; then kill $BACKEND_PID 2>/dev/null || true; fi
    if [ -n "$FRONTEND_PID" ]; then kill $FRONTEND_PID 2>/dev/null || true; fi
    echo -e "${GREEN}Services stopped.${NC}"
}
trap cleanup EXIT INT TERM

# --- Helper: Clear Ports ---
check_and_clear_port() {
    local PORT=$1
    local PID=$(lsof -t -i:$PORT)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}Port $PORT is in use by PID $PID. Killing it...${NC}"
        kill -9 $PID
        sleep 1
    fi
}

echo -e "${CYAN}Starting PINNs4PreCast (Linux)...${NC}"

# 1. Activate Conda
echo -e "${YELLOW}Activating Conda...${NC}"
# Attempt standard locations
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)"

if ! conda activate $CONDA_ENV_NAME; then
    echo -e "${RED}ERROR: Could not activate conda environment '$CONDA_ENV_NAME'${NC}"
    echo -e "Check if the environment exists with: conda env list"
    exit 1
fi

# 2. Clear Ports
check_and_clear_port $BACKEND_PORT
check_and_clear_port $FRONTEND_PORT

# 3. Start Backend
echo -e "\n${GREEN}[1/2] Starting Backend...${NC}"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port $BACKEND_PORT --log-level warning &
BACKEND_PID=$!
sleep 2 # Short wait to check for immediate crash

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}ERROR: Backend failed to start!${NC}"
    exit 1
fi
echo -e "${GREEN}Backend PID: $BACKEND_PID${NC}"

# 4. Start Frontend
echo -e "\n${GREEN}[2/2] Starting Frontend...${NC}"
if [ ! -d "frontend" ]; then
    echo -e "${RED}ERROR: 'frontend' folder not found!${NC}"
    exit 1
fi

cd frontend
# Check if node_modules exists
if [ ! -d "node_modules" ]; then
     echo -e "${RED}ERROR: 'node_modules' missing in frontend/ folder.${NC}"
     echo -e "${YELLOW}Running 'npm install' for you...${NC}"
     npm install
     if [ $? -ne 0 ]; then
        echo -e "${RED}npm install failed.${NC}"
        exit 1
     fi
fi

# Start npm
npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..

# CRITICAL: Wait and verify it didn't crash immediately
echo -e "${YELLOW}Verifying Frontend startup...${NC}"
sleep 4

if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}ERROR: Frontend crashed immediately!${NC}"
    echo -e "Most likely cause: 'next' command not found or syntax error."
    echo -e "Try running manually: cd frontend && npm run dev"
    # Kill backend since we are aborting
    kill $BACKEND_PID
    exit 1
fi

# 5. Success
echo -e "\n${CYAN}PINNs4PreCast LIVE!${NC}"
echo -e "${GREEN}  Backend:  http://localhost:$BACKEND_PORT/docs${NC}"
echo -e "${GREEN}  Frontend: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${YELLOW}  Press Ctrl+C to stop${NC}"

wait $BACKEND_PID $FRONTEND_PID
