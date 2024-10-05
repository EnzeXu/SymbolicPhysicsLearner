#!/bin/bash
# Navigate to the workspace
cd /data2/enze/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t pp_default_1_5 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s pp_default_1_5

  tmux send-keys -t pp_default_1_5 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t pp_default_1_5 "bash jobs/job_20241001_pp_default_1_5.sh" C-m
  echo "Launched jobs/job_20241001_pp_default_1_5.sh on tmux session pp_default_1_5"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t pp"exit" C-m
else
  echo "Session 'pp_default_1_5' already exists. Attaching..."
  tmux attach -t pp_default_1_5
  tmux send-keys -t pp_default_1_5 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t pp_default_1_5 "bash jobs/job_20240401_pp_default_1_5.sh" C-m
fi
