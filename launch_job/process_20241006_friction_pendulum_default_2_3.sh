#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t friction_pendulum_default_2_3 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s friction_pendulum_default_2_3

  tmux send-keys -t friction_pendulum_default_2_3 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t friction_pendulum_default_2_3 "bash jobs/job_20241006_friction_pendulum_default_2_3.sh" C-m
  echo "Launched jobs/job_20241006_friction_pendulum_default_2_3.sh on tmux session friction_pendulum_default_2_3"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t friction_pendulum"exit" C-m
else
  echo "Session 'friction_pendulum_default_2_3' already exists. Attaching..."
  tmux attach -t friction_pendulum_default_2_3
  tmux send-keys -t friction_pendulum_default_2_3 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t friction_pendulum_default_2_3 "bash jobs/job_20240401_friction_pendulum_default_2_3.sh" C-m
fi
