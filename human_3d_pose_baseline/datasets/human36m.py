# references:
# https://github.com/weigq/3d_pose_baseline_pytorch/blob/master/src/datasets/human36m.py
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/linear_model.py#L247

import numpy as np
import torch
from torch.utils.data import Dataset


class Human36M(Dataset):
    def __init__(self, pose_set_2d, pose_set_3d, camera_frame=True):
        """

        Args:
            pose_set_2d (dict[tuple, numpy.array]): 2d pose set.
            pose_set_3d (dict[tuple, numpy.array]): 3d pose set.
            camera_frame (bool, optional): Make this True if pose_set_3d is in camera coordinates. Defaults to True.
        """
        self.poses_2d = []
        self.poses_3d = []

        for key2d in pose_set_2d.keys():
            subj, act, seqname = key2d
            # Keys should be the same if 3d poses are in camera frame.
            key3d = (
                key2d
                if camera_frame
                else (subj, act, "{}.h5".format(seqname.split(".")[0]))
            )

            self.poses_2d.append(pose_set_2d[key2d])
            self.poses_3d.append(pose_set_3d[key3d])

        self.poses_2d = np.vstack(self.poses_2d)
        self.poses_3d = np.vstack(self.poses_3d)

        assert len(self.poses_2d) == len(self.poses_3d)

    def __getitem__(self, idx):
        """Get a pair of 2d and 3d pose.

        Args:
            idx (int): Index of the 2d/3d pose pair to get.

        Returns:
            x (torch.Tensor): 2d pose (model input).
            y (torch.Tensor): 3d pose (model output i.e., label).
        """
        x = torch.from_numpy(self.poses_2d[idx]).float()
        y = torch.from_numpy(self.poses_3d[idx]).float()

        return x, y

    def __len__(self):
        """Return the number of the samples.

        Returns:
            (int): Number of the samples.
        """
        return len(self.poses_2d)
