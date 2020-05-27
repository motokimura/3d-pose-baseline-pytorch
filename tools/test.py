#!/usr/bin/env python3
import json
import os

from _init_path import init_path
from human_3d_pose_baseline.configs import load_config
from human_3d_pose_baseline.datasets import get_dataset
from human_3d_pose_baseline.engines import test_epoch
from human_3d_pose_baseline.models import get_model
from human_3d_pose_baseline.solvers import get_criterion
from human_3d_pose_baseline.utils.cuda import get_device


def main():
    """Evaluate model.
    """

    config = load_config()
    human36m = get_dataset(config)
    model = get_model(config)  # Load weight from config.MODEL.WEIGHT.
    criterion = get_criterion(config)
    device = get_device(config.USE_CUDA)

    # Prepare directory to output evaluation results.
    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=False)

    # Testing.
    print("Evaluating...")
    test_logs = test_epoch(config, model, criterion, human36m, device)

    # Print.
    mpjpe = test_logs["MPJPE"]
    print(f"MPJPE: {mpjpe:.4f}")

    # Log.
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(
            test_logs,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    print(f"Dumped detailed evaluation results to {out_path}.")


if __name__ == "__main__":
    main()
