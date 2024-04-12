#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t sir_default_3 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s sir_default_3

  tmux send-keys -t sir_default_3 "source venv/bin/activate" C-m
  tmux send-keys -t sir_default_3 "bash jobs/job_20240401_sir_default_3_1.sh" C-m
  echo "Launched jobs/job_20240401_sir_default_3_1.sh"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t sir_default_3 "exit" C-m
else
  echo "Session 'sir_default_3' already exists. Attaching..."
  tmux attach -t sir_default_3
  tmux send-keys -t sir_default_3 "source venv/bin/activate" C-m
  tmux send-keys -t sir_default_3 "bash jobs/job_20240401_sir_default_3_1.sh" C-m
fi
