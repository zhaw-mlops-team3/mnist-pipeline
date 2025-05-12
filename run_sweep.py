import wandb
import yaml
from main import main
import os

if "WANDB_API_KEY" not in os.environ:
    raise ValueError("WANDB_API_KEY not set, aborting!!!")
wandb.login(key=os.environ["WANDB_API_KEY"])

sweep_config = None
if sweep_config is None:
    with open("sweep.yaml") as file:
        sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep=sweep_config, project="mnist-pipeline", entity="zhaw-mlops-group3")
wandb.agent(sweep_id, function=main, count=10)
