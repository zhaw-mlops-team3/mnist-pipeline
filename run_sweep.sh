#!/bin/bash
set -e

echo "Logging in to wandb..."
wandb login --relogin "$WANDB_API_KEY"

echo "Creating sweep using wandb Python API..."
SWEEP_ID=$(python3 <<EOF
import wandb
import yaml

with open("sweep.yaml") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep=sweep_config, project="mnist-pipeline", entity="zhaw-mlops-group3")
print(sweep_id)
EOF
)

# Extract the last line only (which is just the ID)
SWEEP_ID=$(echo "$SWEEP_ID" | tail -n 1)

if [ -z "$SWEEP_ID" ]; then
  echo "Could not get sweep ID. ABORTING"
  exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "Starting agent for sweep..."
wandb agent zhaw-mlops-group3/mnist-pipeline/"$SWEEP_ID"
