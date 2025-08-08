import numpy as np


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
