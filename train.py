import wandb


def train(model,train_loader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs + 1):
        running_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        wandb.log({
            "running_loss": running_loss,
            "epoch": epoch}
            )


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss
        loss.backward()
        optimizer.step()

    return running_loss