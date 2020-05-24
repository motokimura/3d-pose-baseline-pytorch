import os

from torch.utils.data import DataLoader

from ..utils import camera_utils, data_utils
from .human36m import Human36M


def get_dataset(config):
    """Get Human3.6M dataset.

    Args:
        config (yacs.config.CfgNode): Configuration.

    Returns:
        (Human36MDatasetHandler): Human3.6M dataset.
    """
    return Human36MDatasetHandler(config)


class Human36MDatasetHandler:
    def __init__(self, config):
        """

        Args:
            config (yacs.config.CfgNode): Configuration.
        """
        # Define actions.
        self.actions = self._get_actions(config)

        # Load Human3.6M camera parameters.
        self.cams = camera_utils.load_cameras(
            os.path.join(config.DATA.HM36M_DIR, "cameras.h5")
        )

        # Load Human3.6M 3d poses.
        print("Loading 3d poses...")
        (
            self.poses_3d_train,
            self.poses_3d_test,
            self.mean_3d,
            self.std_3d,
            self.dim_to_ignore_3d,
            self.dim_to_use_3d,
            self.train_root_positions,
            self.test_root_positions,
        ) = data_utils.read_3d_data(
            self.actions,
            config.DATA.HM36M_DIR,
            self.cams,
            camera_frame=config.DATA.POSE_IN_CAMERA_FRAME,
            predict_14=config.MODEL.PREDICT_14,
        )
        print("Done!")

        # Load Human3.6M 2d poses.
        print("Loading 2d poses...")
        (
            self.poses_2d_train,
            self.poses_2d_test,
            self.mean_2d,
            self.std_2d,
            self.dim_to_ignore_2d,
            self.dim_to_use_2d,
        ) = data_utils.create_2d_data(self.actions, config.DATA.HM36M_DIR, self.cams)
        print("Done!")

        # Create pytorch dataloaders for train and test set.
        self.train_dataloader = self._get_dataloaders(
            config, self.poses_2d_train, self.poses_3d_train, is_train=True
        )

        self.test_dataloader = self._get_dataloaders(
            config, self.poses_2d_test, self.poses_3d_test, is_train=False
        )

    # Private members.

    def _get_actions(self, config):
        actions = config.DATA.ACTIONS
        if len(actions) == 0:
            # If empty, load all actions.
            actions = data_utils.H36M_ACTIONS
        else:
            # Check if the specified actions are valid.
            for act in actions:
                assert act in data_utils.H36M_ACTIONS, f"Unrecognized action: {act}."
        return actions

    def _get_dataloaders(self, config, pose_set_2d, pose_set_3d, is_train):
        # Create pytorch dataset.
        dataset = Human36M(
            pose_set_2d, pose_set_3d, camera_frame=config.DATA.POSE_IN_CAMERA_FRAME
        )

        # Create pytorch dataloader.
        if is_train:
            batch_size = config.LOADER.TRAIN_BATCHSIZE
            num_workers = config.LOADER.TRAIN_NUM_WORKERS
            shuffle = True
        else:
            batch_size = config.LOADER.TEST_BATCHSIZE
            num_workers = config.LOADER.TEST_NUM_WORKERS
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
