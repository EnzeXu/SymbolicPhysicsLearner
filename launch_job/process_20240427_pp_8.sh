#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t pp 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s pp

  tmux send-keys -t pp "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t pp "bash jobs/job_20240427_pp_8.sh" C-m
  echo "Launched jobs/job_20240427_pp_8.sh"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t pp"exit" C-m
else
  echo "Session 'pp' already exists. Attaching..."
  tmux attach -t pp
  tmux send-keys -t pp "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t pp "bash jobs/job_20240427_pp_8.sh" C-m
fi
