import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    camera_positions_estimated: list[np.ndarray],
    camera_positions_ground_truth: list[np.ndarray] | None = None,
) -> None:
    """
    Top-down view (xz)
    """
    if camera_positions_ground_truth:
        cam_x_gt = [c[0] for c in camera_positions_ground_truth]
        cam_z_gt = [c[2] for c in camera_positions_ground_truth]
        plt.plot(
            cam_x_gt,
            cam_z_gt,
            "g-",
            linewidth=2,
            label="Camera Trajectory Ground Truth",
        )

    cam_x = [c[0] for c in camera_positions_estimated]
    cam_z = [c[2] for c in camera_positions_estimated]
    plt.plot(cam_x, cam_z, "b-", linewidth=2, label="Camera Trajectory Estimated")

    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("2D Camera Trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
