import wandb
import yaml
from main import main

sweep_config = None

if sweep_config is None:
    with open("sweep.yaml") as file:
        sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep=sweep_config, project="mnist-pipeline", entity="zhaw-mlops-group3")
wandb.agent(sweep_id, function=main, count=10)
