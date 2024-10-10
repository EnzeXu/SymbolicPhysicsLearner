#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t lorenz_default_1_1 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s lorenz_default_1_1

  tmux send-keys -t lorenz_default_1_1 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t lorenz_default_1_1 "bash jobs/job_20241008_lorenz_default_1_1.sh" C-m
  echo "Launched jobs/job_20241008_lorenz_default_1_1.sh on tmux session lorenz_default_1_1"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t lorenz"exit" C-m
else
  echo "Session 'lorenz_default_1_1' already exists. Attaching..."
  tmux attach -t lorenz_default_1_1
  tmux send-keys -t lorenz_default_1_1 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t lorenz_default_1_1 "bash jobs/job_20240401_lorenz_default_1_1.sh" C-m
fi
