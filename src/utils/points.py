import numpy as np


def apply_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def to_cv2(p_I: np.ndarray) -> np.ndarray:
    """
    Convert format of points to match what cv2 expects

    Args:
        p_I (np.ndarray): (2xN)

    Returns:
        np.ndarray: (Nx1x2)
    """
    p_I = p_I.astype(np.float32)
    N = p_I.shape[1]
    return p_I.T.reshape((N, 1, 2))


def from_cv2(p_I: np.ndarray) -> np.ndarray:
    """
    Convert format of points from cv2 format to (2xN)

    Args:
        p_I (np.ndarray): (Nx1x2)

    Returns:
        np.ndarray: (2xN)
    """
    return p_I.T.reshape(2, -1)
