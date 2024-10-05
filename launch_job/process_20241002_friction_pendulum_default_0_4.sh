#!/bin/bash
# Navigate to the workspace
cd /data2/enze/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t friction_pendulum_default_0_4 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s friction_pendulum_default_0_4

  tmux send-keys -t friction_pendulum_default_0_4 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t friction_pendulum_default_0_4 "bash jobs/job_20241002_friction_pendulum_default_0_4.sh" C-m
  echo "Launched jobs/job_20241002_friction_pendulum_default_0_4.sh on tmux session friction_pendulum_default_0_4"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t friction_pendulum"exit" C-m
else
  echo "Session 'friction_pendulum_default_0_4' already exists. Attaching..."
  tmux attach -t friction_pendulum_default_0_4
  tmux send-keys -t friction_pendulum_default_0_4 "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t friction_pendulum_default_0_4 "bash jobs/job_20240401_friction_pendulum_default_0_4.sh" C-m
fi
