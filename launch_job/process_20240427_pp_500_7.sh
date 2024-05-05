#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t pp_500_7 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s pp_500_7

  tmux send-keys -t pp_500_7 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t pp_500_7 "bash jobs/job_20240427_pp_500_7.sh" C-m
  echo "Launched jobs/job_20240427_pp_500_7.sh on tmux session pp_500_7"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t pp"exit" C-m
else
  echo "Session 'pp_500_7' already exists. Attaching..."
  tmux attach -t pp_500_7
  tmux send-keys -t pp_500_7 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t pp_500_7 "bash jobs/job_20240401_pp_500_7.sh" C-m
fi