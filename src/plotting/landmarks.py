import numpy as np


def plot_landmarks_top_view(
    ax, p_W: np.ndarray, fmt="bx", camera_positions: list[np.ndarray] | None = None
) -> None:
    ax.plot(p_W[0, :], p_W[2, :], fmt)
    if camera_positions is not None:
        for camera_position in camera_positions[-20:]:
            ax.plot(camera_position[0], camera_position[2], "rx")
