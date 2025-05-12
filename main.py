import argparse
import wandb
import torch
from model import Net
import torch.optim as optim
import torch.nn as nn
from train import train
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from evaluate import evaluate
from datetime import datetime
import os

def init_wandb():
    wandb.init(project="mnist-pipeline", entity="zhaw-mlops-group3")
    return wandb.config


def init_device():
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUNNING ON: {device}")
    return device


def main():
    config = init_wandb()
    device = init_device()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    train(model, train_loader, criterion, optimizer, config.epochs, device)
    evaluate(model, test_loader, device)
    
    now = datetime.now()
    time_string = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
    model_config_string = "_".join(f"{k}={v}" for k, v in config.items())
    os.makedirs("trained_models", exist_ok=True)
    model_path = f"trained_models/{model_config_string}__{time_string}.pth"
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(name="mnist-model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    os.remove(model_path)

if __name__ == "__main__":
    main()