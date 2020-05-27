# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/predict_3dpose.py#L305

import numpy as np

from ..utils import data_utils, procrustes


class Human36M_JointErrorEvaluator:
    def __init__(self, human36m, predict_14=False, apply_procrustes_alignment=False):
        """

        Args:
            human36m (Human36MDatasetHandler): Human3.6M dataset.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
            apply_procrustes_alignment (bool, optional): Whether to apply procrustes alignment to the predicted poses.
        """
        self.human36m = human36m
        self.predict_14 = predict_14
        self.apply_procrustes_alignment = apply_procrustes_alignment

        self.n_joints = (
            14 if self.predict_14 else 17
        )  # 17 = predicted 16 joints + root (Hip joint)

        self.reset()

    def reset(self):
        """Remove all samples added so far.
        """
        self.joint_distances = []
        self.actions = []

    def add_samples(self, pred_3d_poses, truth_3d_poses, actions):
        """Add pairs of predicted and ground-truth poses to evaluate.

        Args:
            pred_3d_poses (numpy.array): Predicted 3d poses (normalized). `[batch_size, n_joints, 3]`.
            truth_3d_poses (numpy.array): Ground-truth 3d poses (normalized). `[batch_size, n_joints, 3]`.
            actions (list[str]): Actions to which the poses belong.
        """
        # Compute distances of corresponding joints of pred/truth poses.
        pred = self._preprocess_poses(pred_3d_poses)  # [batch_size, n_joints x 3]
        truth = self._preprocess_poses(truth_3d_poses)  # [batch_size, n_joints x 3]

        if self.apply_procrustes_alignment:
            pred = self._apply_procrustes_alignment(
                sources=pred, targets=truth
            )  # [batch_size, n_joints x 3]

        d = self._compute_joint_distances(pred, truth)  # [batch_size, n_joints]
        self.joint_distances.append(d)

        # Cache action of each frame for per action evaluation.
        self.actions.extend(actions)

    def get_metrics(self):
        """Get evaluation results.

        Returns:
            (dict): evaluation results.
        """
        joint_distances = np.vstack(self.joint_distances)  # [N, n_joints]
        actions = np.array(self.actions)  # [N,]
        assert len(joint_distances) == len(actions)

        # Evaluate joint position errors over all actions.
        mpjpe = np.mean(joint_distances)  # mean per joint position error: float
        pjpe = np.mean(joint_distances, axis=0)  # per joint position error: [n_joints,]
        metrics = {
            "MPJPE": mpjpe,
            "PJPE": pjpe.tolst(),
        }

        # Evaluate joint position error per action.
        for action in data_utils.H36M_ACTIONS:
            mask = actions == action
            if np.sum(mask) == 0:  # In case no sample is found in the action,
                mpjpe = pjpe = -1  # set errors as -1.
                print("Warining: no test sample was found in the action: {action}. ")
            else:
                joint_distances_masked = joint_distances[mask]
                mpjpe = np.mean(joint_distances_masked)
                pjpe = np.mean(joint_distances_masked, axis=0)

            metrics["MPJPE/{}".format(action)] = mpjpe
            metrics["PJPE/{}".format(action)] = pjpe.tolist()

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

    def _apply_procrustes_alignment(self, sources, targets):
        sources_aligned = []

        batch_size = len(sources)
        for i in range(batch_size):
            target = targets[i].reshape(-1, 3)  # [n_joints, 3]
            source = sources[i].reshape(-1, 3)  # [n_joints, 3]
            _, _, T, b, c = procrustes.compute_similarity_transform(
                target, source, compute_optimal_scale=True
            )
            aligned = (b * source.dot(T)) + c
            aligned = aligned.reshape((-1, self.n_joints * 3))  # [1, n_joints x 3]

            sources_aligned.append(aligned)

        return np.vstack(sources_aligned)  # [batch_size, n_joints x 3]

    def _compute_joint_distances(self, pred, truth):
        # Compute Euclidean distance error per joint.
        d_squared = (pred - truth) ** 2  # [batch_size, n_joints x 3]
        d_squared = d_squared.reshape(
            (-1, self.n_joints, 3)
        )  # [batch_size, n_joints, 3]
        d_squared = np.sum(d_squared, axis=2)  # [batch_size, n_joints]
        d = np.sqrt(d_squared)  # [batch_size, n_joints]

        return d
