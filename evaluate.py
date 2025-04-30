import torch
import wandb


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    wandb.log({"test_accuracy": accuracy})
    print(f"Test Accuracy: {accuracy:.2f}%")

