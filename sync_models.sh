#!/bin/bash

HOST="vipde@100.67.131.46"
REMOTE_WIN_DIR="rl_models_exchange"
REMOTE_WSL_DIR="/mnt/c/Users/vipde/rl_models_exchange"
LOCAL_DIR="./models_downloaded"

echo "Creating local directory..."
mkdir -p $LOCAL_DIR

echo "Staging models from WSL to Windows host..."
# Copy checkpoints
ssh -o StrictHostKeyChecking=no $HOST "wsl -u dejar bash -c 'mkdir -p $REMOTE_WSL_DIR; cp -r ~/rl-algoritms/models_fov10/* $REMOTE_WSL_DIR/ 2>/dev/null || echo No checkpoints yet'"
# Copy final model
ssh -o StrictHostKeyChecking=no $HOST "wsl -u dejar bash -c 'cp ~/rl-algoritms/models/ppo_fov10_final.zip $REMOTE_WSL_DIR/ 2>/dev/null || echo No final model yet'"

echo "Downloading models from Windows host..."
scp -o StrictHostKeyChecking=no -r $HOST:$REMOTE_WIN_DIR/* $LOCAL_DIR/

echo "Done! Models are in $LOCAL_DIR"
ls -l $LOCAL_DIR

