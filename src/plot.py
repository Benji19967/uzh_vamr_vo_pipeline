import matplotlib.pyplot as plt
import numpy as np


def plot_tracking(
    I0_keypoints: np.ndarray,
    I1_keypoints: np.ndarray,
):
    x_from = I0_keypoints[0]
    x_to = I1_keypoints[0]
    y_from = I0_keypoints[1]
    y_to = I1_keypoints[1]

    for i in range(x_from.shape[0]):
        plt.plot([x_from[i], x_to[i]], [y_from[i], y_to[i]], "g-", linewidth=3)
    plt.show()
