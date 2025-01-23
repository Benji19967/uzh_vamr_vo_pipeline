import numpy as np


def normalise2DPts(pts):
    """normalises 2D homogeneous points

    Function translates and normalises a set of 2D homogeneous points
    so that their centroid is at the origin and their mean distance from
    the origin is sqrt(2).

    Usage:   [pts_tilde, T] = normalise2dpts(pts)

    Argument:
      pts -  3xN array of 2D homogeneous coordinates

    Returns:
      pts_tilde -  3xN array of transformed 2D homogeneous coordinates.
      T         -  The 3x3 transformation matrix, pts_tilde = T*pts
    """
    pts_euclidean = pts / pts[2, :]

    # Centroid (Euclidean coordinates). Shape (2,): x and y coordinates
    mu = np.mean(pts_euclidean[:2, :], axis=1)
    pts_centered = (pts_euclidean[:2, :].T - mu).T

    sigma = np.sqrt(np.mean(np.sum(pts_centered**2, axis=0)))

    s = np.sqrt(2) / sigma
    T = np.array(
        [
            [s, 0, -s * mu[0]],
            [0, s, -s * mu[1]],
            [0, 0, 1],
        ]
    )

    pts_tilde = T @ pts_euclidean

    return pts_tilde, T
