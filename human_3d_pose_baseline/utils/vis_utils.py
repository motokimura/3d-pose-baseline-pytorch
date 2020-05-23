# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/viz.py

import numpy as np

from .data_utils import H36M_NAMES


def show_3d_pose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    """Visualize a 3d skeleton.

    Args:
        channels (numpy.array): 96x1 vector. The pose to plot.
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): matplotlib axis to draw on.
        lcolor (str, optional): Color for left part of the body. Defaults to "#3498db".
        rcolor (str, optional): Color for right part of the body. Defaults to "#e74c3c".
        add_labels (bool, optional): Whether to add coordinate labels. Defaults to False.
    """

    assert (
        channels.size == len(H36M_NAMES) * 3
    ), "channels should have 96 entries, it has {} instead.".format(channels.size)
    vals = np.reshape(channels, (len(H36M_NAMES), -1))

    # XXX: Joint indices are hard coded.
    I = np.array(
        [0, 1, 2, 0, 6, 7, 0, 12, 13, 14, 13, 17, 18, 13, 25, 26]
    )  # Start points.
    J = np.array(
        [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    )  # End points.
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection array.
    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 750  # Space around the subject.
    # XXX: Assuming index-0 is for root joint.
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    ax.set_aspect("equal")

    # Get rid of the panes (actually make them white).
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane.

    # Get rid of the lines in 3d.
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def show_2d_pose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    """Visualize a 2d skeleton.

    Args:
        channels: 64x1 vector. The pose to plot.
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axis to draw on.
        lcolor (str, optional): Color for left part of the body. Defaults to "#3498db".
        rcolor (str, optional): Color for right part of the body. Defaults to "#e74c3c".
        add_labels (bool, optional): Whether to add coordinate labels. Defaults to False.
    """

    assert (
        channels.size == len(H36M_NAMES) * 2
    ), "channels should have 64 entries, it has {} instead.".format(channels.size)
    vals = np.reshape(channels, (len(H36M_NAMES), -1))

    # XXX: Joint indices are hard coded.
    I = np.array([0, 1, 2, 0, 6, 7, 0, 12, 13, 13, 17, 18, 13, 25, 26])  # Start points.
    J = np.array([1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27])  # End points.
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection array.
    for i in range(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 350  # Space around the subject.
    # XXX: Assuming index-0 is for root joint.
    xroot, yroot = vals[0, 0], vals[0, 1]
    ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Get rid of the ticks and tick labels.
    ax.set_xticks([])
    ax.set_yticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    ax.set_aspect("equal")
    ax.invert_yaxis()
