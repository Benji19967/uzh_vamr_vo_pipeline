import numpy as np


def apply_mask(points: np.ndarray, mask: np.ndarray):
    """
    Apply mask to points if points are not empty.

    Args:
        - points  np.ndarray(Any, N)
        - mask    np.ndarray(N,)
    """

    if points.any():
        return points[:, mask]
    return points


def apply_mask_many(points: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    """
    Apply mask to list of points if points are not empty.

    Args:
        - points  list[np.ndarray(Any, N)]
        - mask    np.ndarray(N,)
    """
    masked = []
    for pts in points:
        masked.append(apply_mask(pts, mask))
    return masked
