#!/bin/bash
# Setup environment and run training
# Designed to be run inside WSL

# Navigate to project root (assuming it's in home)
cd ~/rl-algoritms || exit

# Pull latest changes
git pull

# Setup python path
export PYTHONPATH=$HOME/rl-algoritms/src

# Run training with logging
# Using stdbuf to ensure logs are flushed immediately
stdbuf -oL -eL ./venv/bin/python src/train_fov10.py 2>&1 | tee training.log

