#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t sir_default_12_4 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s sir_default_12_4

  tmux send-keys -t sir_default_12_4 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t sir_default_12_4 "bash jobs/job_20240504_sir_default_12_4.sh" C-m
  echo "Launched jobs/job_20240504_sir_default_12_4.sh on tmux session sir_default_12_4"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t sir"exit" C-m
else
  echo "Session 'sir_default_12_4' already exists. Attaching..."
  tmux attach -t sir_default_12_4
  tmux send-keys -t sir_default_12_4 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t sir_default_12_4 "bash jobs/job_20240401_sir_default_12_4.sh" C-m
fi