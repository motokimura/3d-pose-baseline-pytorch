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
        self.actions = []

        for key2d in pose_set_2d.keys():
            subj, act, seqname = key2d
            # Keys should be the same if 3d poses are in camera frame.
            key3d = (
                key2d
                if camera_frame
                else (subj, act, "{}.h5".format(seqname.split(".")[0]))
            )

            poses_2d = pose_set_2d[key2d]  # [n, 16 x 2]
            poses_3d = pose_set_3d[key3d]  # [n, n_joints x 3]
            assert len(poses_2d) == len(poses_3d)
            actions = [act] * len(poses_2d)  # [n,]

            self.poses_2d.append(poses_2d)
            self.poses_3d.append(poses_3d)
            self.actions.extend(actions)

        self.poses_2d = np.vstack(self.poses_2d)  # [N, 16 x 2]
        self.poses_3d = np.vstack(self.poses_3d)  # [N, n_joints x 3]
        self.actions = np.array(self.actions)  # [N,]

        assert len(self.poses_2d) == len(self.poses_3d) == len(self.actions)

    def __getitem__(self, idx):
        """Get a set of 2d pose, 3d pose, and action.

        Args:
            idx (int): Index of the 2d/3d pose pair to get.

        Returns:
            (dict): a set of 2d pose, 3d pose, and action.
            pose_2d (torch.Tensor): 2d pose (model input).
            pose_3d (torch.Tensor): 3d pose (model output i.e., label).
            action (str): Action to which the pose pair belongs.
        """
        pose_2d = torch.from_numpy(self.poses_2d[idx]).float()
        pose_3d = torch.from_numpy(self.poses_3d[idx]).float()
        action = self.actions[idx]

        return {"pose_2d": pose_2d, "pose_3d": pose_3d, "action": action}

    def __len__(self):
        """Return the number of the samples.

        Returns:
            (int): Number of the samples.
        """
        return len(self.poses_2d)
