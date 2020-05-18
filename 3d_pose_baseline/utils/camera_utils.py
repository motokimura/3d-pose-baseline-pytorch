# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/cameras.py

import os

import h5py
import numpy as np


def transform_camera_to_world(P, R, T):
    """Transform points from camera to world coordinates.

    Args:
        P (numpy.array): Nx3 3d points in camera coordinates.
        R (numpy.array): Camera rotation matrix.
        T (numpy.array): Camera translation vector.

    Returns:
        X (numpy.array): Nx3 3d points in world coordinates.
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X = R.T @ (P.T) + T  # rotate and translate
    return X.T


def transform_world_to_camera(P, R, T):
    """Transform points from world to camera coordinates.

    Args:
        P (numpy.array): Nx3 3d points in world coordinates.
        R (numpy.array): Camera rotation matrix.
        T (numpy.array): Camera translation vector.

    Returns:
        X (numpy.array): Nx3 3d points in camera coordinates.
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X = R @ (P.T - T)  # rotate and translate
    return X.T


def project_to_camrea(P, R, T, f, c, k, p):
    """Project points from 3d to 2d using camera parameters
    including radial and tangential distortions.

    Args:
        P (numpy.array): Nx3 3d points in world coordinates.
        R (numpy.array): 3x3 Camera rotation matrix.
        T (numpy.array): 3x1 Camera translation parameters.
        f (numpy.array): 2x1 Camera focal length.
        c (numpy.array): 2x1 Camera center.
        k (numpy.array): 3x1 Camera radial distortion coefficients.
        p (numpy.array): 2x1 Camera tangential distortion coefficients.
    Returns:
        p (numpy.array): Nx2 2d points in pixel space.
        d (numpy.array): 1xN depth of each point in camera space.
        radial (numpy.array): 1xN radial distortion per point.
        tan (numpy.array): 1xN tangential distortion per point.
        r2 (numpy.array): 1xN squared radius of the projected points before distortion.
    """
    N = P.shape[0]

    X = transform_world_to_camera(P, R, T)  # Nx3
    X = X.T  # 3xN
    d = X[2, :]  # Depth.
    XX = X[:2, :] / d  # 2xN

    # Radial distorsion term
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    # Tangential distorsion term.
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]
    # Apply the distorsions.
    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    # Project to camera.
    projected = f * XXX + c
    projected = projected.T  # Nx2

    return projected, d, radial, tan, r2


def load_camera_params(hf, base_path):
    """Load h36m camera parameters.

    Args:
        hf (file object): HDF5 open file with h36m cameras data
        path (str): Path or key inside hf to the camera we are interested in.

    Returns:
        R (numpy.array): 3x3 Camera rotation matrix.
        T (numpy.array): 3x1 Camera translation parameters.
        f (numpy.array): 2x1 Camera focal length.
        c (numpy.array): 2x1 Camera center.
        k (numpy.array): 3x1 Camera radial distortion coefficients.
        p (numpy.array): 2x1 Camera tangential distortion coefficients.
        name (str): String with camera id.
    """
    R = hf[os.path.join(base_path, "R")][:]
    R = R.T

    T = hf[os.path.join(base_path, "T")][:]
    f = hf[os.path.join(base_path, "f")][:]
    c = hf[os.path.join(base_path, "c")][:]
    k = hf[os.path.join(base_path, "k")][:]
    p = hf[os.path.join(base_path, "p")][:]

    name = hf[os.path.join(base_path, "Name")][:]
    name = "".join(chr(item) for item in name)

    return R, T, f, c, k, p, name


def load_cameras(camera_h5_path, subjects=[1, 5, 6, 7, 8, 9, 11]):
    """Load camera parameters from camera.h5 of Human3.6M

    Args:
        camera_h5_path (str): Path to hdf5 file of Human3.6M camera params
        subjects (list[int], optional): List of ints representing
        the subject IDs for which cameras are requested.
        Defaults to [1, 5, 6, 7, 8, 9, 11].

    Returns:
        cams (dict[tuple, tuple]): Dictionary of 4 tuples per subject ID
        containing its camera parameters for the 4 Human3.6M cameras.
    """
    cams = {}

    with h5py.File(camera_h5_path, 'r') as hf:
        for subj in subjects:
            for cam_idx in range(1, 5):  # Human3.6M has 4 cameras (#1~5) for each subject.
                base_path = f"subject{subj}/camera{cam_idx}"
                cams[(subj, cam_idx)] = load_camera_params(hf, base_path)

    return cams
