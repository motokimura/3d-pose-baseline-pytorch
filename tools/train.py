#!/usr/bin/env python3
import torch
from tqdm import tqdm

from _init_path import init_path
from human_3d_pose_baseline.configs import load_config
from human_3d_pose_baseline.datasets import get_dataset
from human_3d_pose_baseline.engines import test_epoch, train_epoch
from human_3d_pose_baseline.models import get_model
from human_3d_pose_baseline.solvers import (
    get_criterion,
    get_lr_scheduler,
    get_optimizer,
)
from human_3d_pose_baseline.utils.cuda import get_device


def main():
    """Train model.
    """

    config = load_config()
    human36m = get_dataset(config)
    model = get_model(config)
    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)
    device = get_device(config.USE_CUDA)

    for epoch in range(config.SOLVER.EPOCHS):
        print(f"Epoch: {epoch}")

        print("Training...")
        train_logs = train_epoch(
            config, model, criterion, optimizer, lr_scheduler, human36m, device
        )

        print("Testing...")
        test_logs = test_epoch(config, model, criterion, human36m, device)


if __name__ == "__main__":
    main()
