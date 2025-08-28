import matplotlib.pyplot as plt
import numpy as np

DPI = 227  # Mac M1


def plot_tracking(
    I0_keypoints: np.ndarray,
    I1_keypoints: np.ndarray,
    figsize_pixels_x: int | None = None,
    figsize_pixels_y: int | None = None,
):
    """
    Plot keypoint tracking from one image to the next.

    Args:
        I0_keypoints np.ndarray(2,N): p=(x,y)
        I1_keypoints np.ndarray(2,N): p=(x,y)
        figsize_pixels_x: x size of figure in pixels
        figsize_pixels_y: y size of figure in pixels
    """
    x_from = I0_keypoints[0]
    x_to = I1_keypoints[0]
    y_from = I0_keypoints[1]
    y_to = I1_keypoints[1]

    if figsize_pixels_x and figsize_pixels_y:
        plt.figure(figsize=(figsize_pixels_x / DPI, figsize_pixels_y / DPI), dpi=DPI)
        ax = plt.gca()
        ax.set_xlim([0, figsize_pixels_x + 1])  # type: ignore
        ax.set_ylim([0, figsize_pixels_y + 1])  # type: ignore
    plt.gca().invert_yaxis()  # because p=(x, y) of keypoints are given for origin at top left corner  # type: ignore
    for i in range(x_from.shape[0]):
        plt.plot([x_from[i], x_to[i]], [y_from[i], y_to[i]], "g-", linewidth=1)
    # plt.show()
    plt.pause(0.05)
