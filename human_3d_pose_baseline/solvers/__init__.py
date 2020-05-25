import torch.nn as nn
import torch.optim as optim

from ..utils.cuda import get_device


def get_criterion(config):
    """Get criterion (loss) to train models.

    Args:
        config (yacs.config.CfgNode): Configuration.

    Returns:
        (torch.nn.Module): Loss function.
    """
    criterion = nn.MSELoss(reduction="mean")

    device = get_device(config.USE_CUDA)
    criterion.to(device)

    return criterion


def get_lr_scheduler(config, optimizer):
    """Get learning rate scheduler to train models.

    Args:
        config (yacs.config.CfgNode): Configuration.
        optimizer (torch.optimizer): Optimizer.

    Returns:
        (torch.lr_scheduler): Learning rate scheduler.
    """
    gamma = config.SOLVER.LR_DECAY_GAMMA
    decay_step = config.SOLVER.LR_DECAY_STEP
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: gamma ** (step / decay_step)
    )
    return lr_scheduler


def get_optimizer(config, model):
    """Get optimizer to train models.

    Args:
        config (yacs.config.CfgNode): Configuration.
        model (torch.nn.Module): Model to train.

    Returns:
        (torch.optimizer): Optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=config.SOLVER.LR)
    return optimizer
