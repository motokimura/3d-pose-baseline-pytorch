import torch
from tqdm import tqdm


def train_epoch(config, model, criterion, optimizer, lr_scheduler, human36m, device):
    """Train the model for an epoch.

    Args:
        config ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        lr_scheduler ([type]): [description]
        human36m ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    model.train()

    sum_loss, num_samples = 0, 0

    for data, target in tqdm(human36m.train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_size = len(data)
        sum_loss += loss.item() * batch_size
        num_samples += batch_size

    average_loss = sum_loss / num_samples

    return {"loss": average_loss}


def test_epoch(config, model, criterion, human36m, device):
    """Evaluate the model.

    Args:
        config ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        human36m ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    model.eval()

    sum_loss, num_samples = 0, 0

    with torch.no_grad():
        for data, target in tqdm(human36m.test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            batch_size = len(data)
            sum_loss += loss.item() * batch_size
            num_samples += batch_size

    average_loss = sum_loss / num_samples

    return {"loss": average_loss}