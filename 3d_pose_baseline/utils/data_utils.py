# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/data_utils.py

import copy
import os
from glob import glob

import h5py
import numpy as np

from .camera_utils import project_to_camrea
from .camera_utils import transform_world_to_camera as transform_world_to_camera_base

# Human3.6m IDs for training and testing.
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

# Joints in Human3.6M;
# data has 32 joints, but only 17 that move.
H36M_NAMES = [""] * 32
H36M_NAMES[0] = "Hip"
H36M_NAMES[1] = "RHip"
H36M_NAMES[2] = "RKnee"
H36M_NAMES[3] = "RFoot"
H36M_NAMES[6] = "LHip"
H36M_NAMES[7] = "LKnee"
H36M_NAMES[8] = "LFoot"
H36M_NAMES[12] = "Spine"
H36M_NAMES[13] = "Thorax"
H36M_NAMES[14] = "Neck/Nose"
H36M_NAMES[15] = "Head"
H36M_NAMES[17] = "LShoulder"
H36M_NAMES[18] = "LElbow"
H36M_NAMES[19] = "LWrist"
H36M_NAMES[25] = "RShoulder"
H36M_NAMES[26] = "RElbow"
H36M_NAMES[27] = "RWrist"


def load_data(data_dir, subjects, actions):
    """Load 3d ground truth from disk, and puts it in an easy-to-access dictionary.

    Args:
        data_dir (str): Path to where to load the data from.
        subjects (list[int]): Subjects whose data will be loaded.
        actions (list[str]): Actions to load.

    Returns:
        data (dict[tuple, numpy.array]): Directionary with
        keys k=(subjects, actions, seqname) and
        values v=(nx(32x3) matrix of 3d ground truth).
    """

    data = {}
    for subj in subjects:
        for act in actions:
            print(f"reading subject {subj}, action {act}...")

            path = os.path.join(data_dir, f"S{subj}/MyPoses/3D_positions/{act}*.h5")

            fnames = glob(path)

            loaded_seqs = 0
            for fname in fnames:
                seqname = os.path.basename(fname)

                # This makes sure SittingDown is not loaded when Sitting is required.
                if act == "Sitting" and seqname.startswith("SittingDown"):
                    continue

                if seqname.startswith(act):
                    print(fname)
                    loaded_seqs += 1

                    with h5py.File(fname, "r") as f:
                        poses = f["3D_positions"][:]

                    poses = poses.T  # [N, 96]
                    data[(subj, act, seqname)] = poses

            assert (
                loaded_seqs == 2
            ), f"Expecting 2 sequences, but found {loaded_seqs} instead."

    return data


def transform_world_to_camera(pose_set, cams, ncams=4):
    """Transform 3d poses from world coordinate to camera coordinate.

    Args:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d poses.
        cams (dict[tuple, tuple]): Dictionary with cameras.
        ncams (int, optional): Number of cameras per subject. Defaults to 4.

    Returns:
        t3d_camera (dict[tuple, numpy.array]): Dictionary with 3d poses in camera coordinate.
    """
    t3d_camera = {}
    for t3dk in sorted(pose_set.keys()):
        subj, act, seqname = t3dk
        t3d_world = pose_set[t3dk]  # nx(32x3)
        t3d_world = t3d_world.reshape((-1, 3))  # (nx32)x3

        for cam_idx in range(1, ncams + 1):
            R, T, f, c, k, p, name = cams[(subj, cam_idx)]
            camera_coord = transform_world_to_camera_base(t3d_world, R, T)
            camera_coord = camera_coord.reshape((-1, len(H36M_NAMES) * 3))  # nx(32x3)

            base_seqname = seqname[:-3]  # remove ".h5"
            sname = f"{base_seqname}.{name}.h5"  # e.g., "Waiting 1.58860488.h5"
            t3d_camera[(subj, act, sname)] = camera_coord

    return t3d_camera


def postprocess_3d(pose_set):
    """Centerize 3d joint points around root joint.

    Args:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d data.

    Returns:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d data centred around root (center hip) joint.
        root_positions (dict[tuple, numpy.array]): Dictionary with the original 3d position of each pose.
    """
    root_positions = {}
    for k in sorted(pose_set.keys()):
        poses = pose_set[k]  # nx(32x3)

        # Keep track of global position.
        root_begin = H36M_NAMES.index("Hip") * 3
        root_position = copy.deepcopy(poses[:, root_begin : root_begin + 3])  # nx3

        # Centerize around root.
        poses = poses - np.tile(root_position, [1, len(H36M_NAMES)])

        pose_set[k] = poses
        root_positions[k] = root_position

    return pose_set, root_positions


def compute_normalization_stats(data, dim, predict_14=False):
    """Compute normalization statistics: mean, std, dimensions to use and ignore.

    Args:
        data (numpy.array): nxd array of poses
        dim (int): Dimensionality of the pose. 2 or 3.
        predict_14 (bool, optional): Whether to use only 14 joints. Defaults to False.

    Returns:
        data_mean (numpy.array): Vector with the mean of the data.
        data_std (numpy.array): Vector with the standard deviation of the data.
        dim_to_ignore (numpy.array): List of dimensions not used in the model.
        dim_to_use (numpy.array): List of dimensions used in the model.
    """
    assert dim in [2, 3], "dim must be 2 or 3."

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    if dim == 2:
        # Get dimensions of 16 2d points to use.
        dim_to_ignore = np.where(
            np.array([x in ["", "Neck/Nose"] for x in H36M_NAMES])
        )[0]
        dim_to_ignore = np.sort(np.hstack([dim_to_ignore * 2, dim_to_ignore * 2 + 1]))
        dim_to_use = np.delete(np.arange(len(H36M_NAMES) * 2), dim_to_ignore)
    else:  # dim == 3
        # Get dimensions of 16 (or 14) 3d points to use.
        if predict_14:
            dim_to_ignore = np.where(
                np.array([x in ["", "Hip", "Spine", "Neck/Nose"] for x in H36M_NAMES])
            )[0]
        else:  # predict 16 points
            dim_to_ignore = np.where(np.array([x in ["", "Hip"] for x in H36M_NAMES]))[
                0
            ]

        dim_to_ignore = np.sort(
            np.hstack([dim_to_ignore * 3, dim_to_ignore * 3 + 1, dim_to_ignore * 3 + 2])
        )
        dim_to_use = np.delete(np.arange(len(H36M_NAMES) * 3), dim_to_ignore)

    return data_mean, data_std, dim_to_ignore, dim_to_use


def normalize_data(data, data_mean, data_std, dim_to_use):
    """Normalize poses in the dictionary.

    Args:
        data (dict[tuple, numpy.array]): Dictionary with the poses.
        data_mean (numpy.array): Vector with the mean of the data.
        data_std (numpy.array): Vector with the std of the data.
        dim_to_use (numpy.array): Dimensions to keep in the data.

    Returns:
        data_normalized (dict[tuple, numpy.array]): Dictionary with same keys as data, but values have been normalized.
    """
    data_normalized = {}

    for key in sorted(data.keys()):
        data[key] = data[key][:, dim_to_use]  # remove joints to ignore
        mu = data_mean[dim_to_use]
        sigma = data_std[dim_to_use]
        data_normalized[key] = np.divide((data[key] - mu), sigma)

    return data_normalized


def read_3d_data(actions, data_dir, cams, camera_frame=True, predict_14=False):
    """Load 3d poses, zero-centred and normalized.

    Args:
        actions (list[str]): Actions to load.
        data_dir (str): Directory where the data can be loaded from.
        cams (dict[tuple, tuple]): Dictionary with camera parameters.
        camera_frame (bool, optional): Whether to convert the data to camera coordinates. Defaults to True.
        predict_14 (bool, optional): Whether to predict only 14 joints. Defaults to False.

    Returns:
        train_set (dict[tuple, numpy.array]): Dictionary with loaded 3d poses for training.
        test_set (dict[tuple, numpy.array]): Dictionary with loaded 3d poses for testing.
        data_mean (numpy.array): Vector with the mean of the 3d training data.
        data_std (numpy.array): Vector with the standard deviation of the 3d training data.
        dim_to_ignore (list[int]): List with the dimensions not to predict.
        dim_to_use (list[int]): List with the dimensions to predict.
        train_root_positions (dict[tuple, numpy.array]): Dictionary with the 3d positions of the root in train set.
        test_root_positions (dict[tuple, numpy.array]: Dictionary with the 3d positions of the root in test set.
    """
    # Load 3d data.
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions)

    if camera_frame:
        train_set = transform_world_to_camera(train_set, cams)
        test_set = transform_world_to_camera(test_set, cams)

    # Centering around root (center hip joint).
    train_set, train_root_positions = postprocess_3d(train_set)
    test_set, test_root_positions = postprocess_3d(test_set)

    # Compute normalization statistics.
    train_concat = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = compute_normalization_stats(
        train_concat, dim=3, predict_14=predict_14
    )

    # Divide every dimension independently.
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return (
        train_set,
        test_set,
        data_mean,
        data_std,
        dim_to_ignore,
        dim_to_use,
        train_root_positions,
        test_root_positions,
    )


def project_to_camreas(pose_set, cams, ncams=4):
    """Project 3d poses using camera parameters.

    Args:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d poses.
        cams (dict[tuple, tuple]): Dictionary with cameras.
        ncams (int, optional): Number of cameras per subject. Defaults to 4.

    Returns:
        t2d (dict[tuple, numpy.array]): Dictionary with projected 2d poses.
    """
    t2d = {}

    for t3dk in sorted(pose_set.keys()):
        subj, act, seqname = t3dk
        t3d = pose_set[t3dk]  # nx(32x3)
        t3d = t3d.reshape((-1, 3))  # (nx32)x3

        for cam_idx in range(1, ncams + 1):
            R, T, f, c, k, p, name = cams[(subj, cam_idx)]
            pts2d, _, _, _, _ = project_to_camrea(t3d, R, T, f, c, k, p)  # (nx32)x2
            pts2d = pts2d.reshape((-1, len(H36M_NAMES) * 2))  # nx(32x2)

            base_seqname = seqname[:-3]  # remove ".h5"
            sname = f"{base_seqname}.{name}.h5"  # e.g., "Waiting 1.58860488.h5"
            t2d[(subj, act, sname)] = pts2d

    return t2d


def create_2d_data(actions, data_dir, cams):
    """Create 2d poses by projecting 3d poses with the corresponding camera parameters,
    and also normalize the 2d poses.

    Args:
        actions (list[str]): Actions to load.
        data_dir (str): Directory where the data can be loaded from.
        cams (dict[tuple, tuple]): Dictionary with camera parameters.

    Returns:
        train_set (dict[tuple, numpy.array]): Dictionary with loaded 2d poses for training.
        test_set (dict[tuple, numpy.array]): Dictionary with loaded 2d poses for testing.
        data_mean (numpy.array): Vector with the mean of the 2d training data.
        data_std (numpy.array): Vector with the standard deviation of the 2d training data.
        dim_to_ignore (list[int]): List with the dimensions not to predict.
        dim_to_use (list[int]): List with the dimensions to predict.
    """
    # Load 3d data.
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions)

    train_set = project_to_camreas(train_set, cams)
    test_set = project_to_camreas(test_set, cams)

    # Compute normalization statistics.
    train_concat = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = compute_normalization_stats(
        train_concat, dim=2, predict_14=False
    )

    # Divide every dimension independently.
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def unnormalize_data(data, data_mean, data_std, dim_to_ignore):
    """Un-normalize poses whose mean has been substracted and that has been divided by
    standard deviation. Returned array has mean values at ignored dimensions.

    Args:
        data (numpy.array): nxd array to unnormalize
        data_mean (numpy.array): Vector with the mean of the data.
        data_std (numpy.array): Vector with the std of the data.
        dim_to_ignore (numpy.array): Dimensions that were removed from the original data.

    Returns:
        data_unnormalized (numpy.array): unnormalized array
    """
    N = data.shape[0]  # Batch size.
    D = data_mean.shape[0]  # Dimensionality.
    data_unnormalized = np.zeros((N, D), dtype=np.float32)  # NxD

    dim_to_use = [d for d in range(D) if d not in dim_to_ignore]
    data_unnormalized[:, dim_to_use] = data

    # unnormalize with mean and std
    sigma = data_std.reshape((1, D))  # 1xD
    sigma = np.repeat(sigma, N, axis=0)  # NxD
    mu = data_mean.reshape((1, D))  # 1xD
    mu = np.repeat(mu, N, axis=0)  # NxD
    data_unnormalized = np.multiply(data_unnormalized, sigma) + mu

    return data_unnormalized
