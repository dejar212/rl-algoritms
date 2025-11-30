#!/bin/bash
# start_training.sh
# Universal launcher for WSL training
# Usage: ./start_training.sh

# Absolute paths
PROJECT_DIR="/home/dejar/rl-algoritms"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
SCRIPT="$PROJECT_DIR/src/train_fov10.py"
LOG_FILE="$PROJECT_DIR/training_final.log"

# Ensure we are in the right directory
cd "$PROJECT_DIR" || { echo "Directory $PROJECT_DIR not found"; exit 1; }

# Setup Environment
export PYTHONPATH="$PROJECT_DIR/src"

echo "=== Starting Training ==="
echo "Working Dir: $(pwd)"
echo "Log File:    $LOG_FILE"

# Kill any existing instances to prevent conflicts
pkill -f train_fov10.py
echo "Old processes killed."

# Run in background with setsid and nohup to survive SSH disconnect
# Redirect both stdout and stderr to log file
# Redirect stdin from /dev/null to prevent hanging
setsid nohup "$VENV_PYTHON" "$SCRIPT" > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "Training started with PID: $PID"
echo "You can check status with: ps aux | grep $PID"
echo "To follow logs: tail -f $LOG_FILE"
echo "========================="

