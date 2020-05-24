import torch.nn as nn
import torch.optim as optim

from ..utils.cuda import get_device


def get_criterion(config):
    """[summary]

    Args:
        config ([type]): [description]

    Returns:
        [type]: [description]
    """
    criterion = nn.MSELoss(reduction="mean")

    device = get_device(config.USE_CUDA)
    criterion.to(device)

    return criterion


def get_lr_scheduler(config, optimizer):
    """[summary]

    Args:
        config ([type]): [description]
        optimizer ([type]): [description]

    Returns:
        [type]: [description]
    """
    gamma = config.SOLVER.LR_DECAY_GAMMA
    decay_step = config.SOLVER.LR_DECAY_STEP
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: gamma ** (step / decay_step)
    )
    return lr_scheduler


def get_optimizer(config, model):
    """[summary]

    Args:
        config ([type]): [description]
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    optimizer = optim.Adam(model.parameters(), lr=config.SOLVER.LR)
    return optimizer
