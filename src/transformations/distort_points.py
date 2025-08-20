import numpy as np


def distort_points(x: np.ndarray, D: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points on the image plane.

    Args:
        x: 2d points (2xN)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)

    Returns:
        distorted points: 2d points (2xN)
    """
    u_0 = K[0, 2]
    v_0 = K[1, 2]
    k_1, k_2 = D[0], D[1]

    xp = x[0, :] - u_0
    yp = x[1, :] - v_0

    r_squared = xp**2 + yp**2
    mult = 1 + k_1 * r_squared + k_2 * r_squared**2

    xpp = u_0 + mult * xp
    ypp = v_0 + mult * yp

    distorted_points = np.stack([xpp, ypp], axis=0)

    return distorted_points
