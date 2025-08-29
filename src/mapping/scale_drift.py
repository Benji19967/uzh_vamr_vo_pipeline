import numpy as np


def compute_scale_drift(
    camera_positions_estimated: list[np.ndarray],
    camera_positions_ground_truth: list[np.ndarray],
):
    """
    Computes scale drift over time for camera positions as a list of (3,) np.ndarrays.

    Args:
        camera_positions_estimated: List of (3,) numpy arrays (ground truth)
        camera_positions_ground_truth: List of (3,) numpy arrays (estimated)

    Returns:
        scale_ratios: List of scale ratios per frame (len N-1)
    """
    scale_ratios = []

    for i in range(1, len(camera_positions_estimated)):
        gt_dist = np.linalg.norm(
            camera_positions_ground_truth[i] - camera_positions_ground_truth[i - 1]
        )
        est_dist = np.linalg.norm(
            camera_positions_estimated[i] - camera_positions_estimated[i - 1]
        )

        if gt_dist > 1e-6:
            scale = est_dist / gt_dist
        else:
            scale = 1.0  # No ground-truth movement, assume scale 1

        scale_ratios.append(scale)

    return scale_ratios
