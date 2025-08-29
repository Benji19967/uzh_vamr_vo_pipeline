import matplotlib.pyplot as plt
import numpy as np

from src.mapping.scale_drift import compute_scale_drift


def plot_scale_drift(
    camera_positions_estimated: list[np.ndarray],
    camera_positions_ground_truth: list[np.ndarray],
) -> None:
    """
    Plot the reprojection error over time/frames
    """
    scale_drift = compute_scale_drift(
        camera_positions_estimated, camera_positions_ground_truth
    )
    plt.plot(scale_drift, "b-", linewidth=2, label="Scale Drift")

    plt.xlabel("Frame id")
    plt.ylabel("Scale (Estimated / Ground Truth)")
    plt.title("Scale Drift Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
