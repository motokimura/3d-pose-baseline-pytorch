#!/usr/bin/env python3
import os

import torch
from tensorboardX import SummaryWriter
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

    # Prepare directory to output training logs and model weights.
    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=False)
    # Prepare tensorboard logger.
    tblogger = SummaryWriter(out_dir)

    mpjpe_lowest = 1e10

    for epoch in range(config.SOLVER.EPOCHS):
        print()

        lr = lr_scheduler.get_last_lr()[0]
        tblogger.add_scalar("lr", lr, epoch)
        print(f"epoch: {epoch}, lr: {lr : .10f}")

        # Training.
        print("Training...")
        train_logs = train_epoch(
            config, model, criterion, optimizer, lr_scheduler, human36m, device
        )

        train_loss = train_logs["loss"]
        tblogger.add_scalar("train/loss", train_loss, epoch)
        print(f"train_loss: {train_loss : .8f}")

        # Testing.
        print("Testing...")
        test_logs = test_epoch(config, model, criterion, human36m, device)

        test_loss = test_logs["loss"]
        mpjpe = test_logs["mean_per_joint_position_error"]
        tblogger.add_scalar("test/loss", test_loss, epoch)
        tblogger.add_scalar("test/mpjpe", mpjpe, epoch)
        print(f"test_loss: {test_loss : .8f}, mpjpe[mm]: {mpjpe : .4f}")

        # Save model weight if lowest error is updated.
        if mpjpe < mpjpe_lowest:
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
            mpjpe_lowest = mpjpe
            print("Test score updated! Saved model weight.")


if __name__ == "__main__":
    main()
