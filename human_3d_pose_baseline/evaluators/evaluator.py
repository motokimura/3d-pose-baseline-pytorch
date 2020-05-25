# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/predict_3dpose.py#L305

import numpy as np

from ..utils import data_utils


class Human36M_JointErrorEvaluator:
    def __init__(self, human36m, predict_14=False):
        """

        Args:
            human36m (Human36MDatasetHandler): Human3.6M dataset.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        self.human36m = human36m
        self.predict_14 = predict_14

        self.n_joints = (
            14 if self.predict_14 else 17
        )  # 17 = predicted 16 joints + root (Hip joint)

        self.reset()

    def reset(self):
        """Remove all samples added so far.
        """
        self.joint_distances = np.zeros(shape=(0, self.n_joints))

    def add_samples(self, pred_3d_poses, truth_3d_poses):
        """Add pairs of predicted and ground-truth poses to evaluate.

        Args:
            pred_3d_poses (numpy.array): Predicted 3d poses (normalized). `[batch_size, n_joints, 3]`.
            truth_3d_poses (numpy.array): Ground-truth 3d poses (normalized). `[batch_size, n_joints, 3]`.
        """
        pred = self._preprocess_poses(pred_3d_poses)  # [batch_size, n_joints x 3]
        truth = self._preprocess_poses(truth_3d_poses)  # [batch_size, n_joints x 3]
        d = self._compute_joint_distances(pred, truth)  # [batch_size, n_joints]

        self.joint_distances = np.vstack([self.joint_distances, d])  # [N, n_joints]

    def get_metrics(self):
        """Get evaluation results.

        Returns:
            (dict): evaluation results.
        """
        mean_per_joint_position_error = np.mean(self.joint_distances)  # float
        per_joint_position_error = np.mean(self.joint_distances, axis=0)  # [n_joints,]

        metrics = {
            "mean_per_joint_position_error": mean_per_joint_position_error,
            "per_joint_position_error": per_joint_position_error,
        }
        return metrics

    def _preprocess_poses(self, poses_3d):
        mean_3d = self.human36m.mean_3d
        std_3d = self.human36m.std_3d
        dim_to_ignore_3d = self.human36m.dim_to_ignore_3d
        dim_to_use_3d = self.human36m.dim_to_use_3d

        # Unnormalize 3d poses.
        poses = data_utils.unnormalize_data(
            poses_3d, mean_3d, std_3d, dim_to_ignore_3d
        )  # [batch_size, 32 x 3]

        # Keep only the relevant joints.
        dim_to_keep = (
            dim_to_use_3d
            if self.predict_14
            else np.hstack([np.arange(3), dim_to_use_3d])
            # Add root (Hip joint) if the model predicts 16 joints.
            # XXX: Assuming the first 3 values represent root joint 3d position.
        )
        poses = poses[:, dim_to_keep]  # [batch_size, n_joints x 3]

        return poses

    def _compute_joint_distances(self, pred, truth):
        # Compute Euclidean distance error per joint.
        d_squared = (pred - truth) ** 2  # [batch_size, n_joints x 3]
        d_squared = d_squared.reshape(
            (-1, self.n_joints, 3)
        )  # [batch_size, n_joints, 3]
        d_squared = np.sum(d_squared, axis=2)  # [batch_size, n_joints]
        d = np.sqrt(d_squared)  # [batch_size, n_joints]

        return d
