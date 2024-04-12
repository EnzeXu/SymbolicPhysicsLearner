#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t lorenz_default_2 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s lorenz_default_2

  tmux send-keys -t lorenz_default_2 "source venv/bin/activate" C-m
  tmux send-keys -t lorenz_default_2 "bash jobs/job_20240401_lorenz_default_2_7.sh" C-m
  echo "Launched jobs/job_20240401_lorenz_default_2_7.sh"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t lorenz_default_2 "exit" C-m
else
  echo "Session 'lorenz_default_2' already exists. Attaching..."
  tmux attach -t lorenz_default_2
  tmux send-keys -t lorenz_default_2 "source venv/bin/activate" C-m
  tmux send-keys -t lorenz_default_2 "bash jobs/job_20240401_lorenz_default_2_7.sh" C-m
fi
