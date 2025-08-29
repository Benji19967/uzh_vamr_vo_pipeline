import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(camera_positions: list[np.ndarray]) -> None:
    """
    Top-down view (xz)
    """
    cam_x = [c[0] for c in camera_positions]
    cam_z = [c[2] for c in camera_positions]

    plt.plot(cam_x, cam_z, "b-", linewidth=2, label="Camera Trajectory")
    plt.scatter(cam_x, cam_z, c="k", s=30)  # optional: mark camera positions

    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("2D Camera Trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
