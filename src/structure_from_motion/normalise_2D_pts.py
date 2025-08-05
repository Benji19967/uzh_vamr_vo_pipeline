import numpy as np

"""TODO: notes on naming
- [1, 2, 1] is hom for a 2D point, but in canonical form (last coordinate is 1) 
(also called normalized form but different 'normalized' than we use in course)

- [2, 4, 2] is hom for a 2D point
"""


def normalise2DPts(p_P_hom):
    """normalises 2D homogeneous points

    Function translates and normalises a set of 2D homogeneous points
    so that their centroid is at the origin and their mean distance from
    the origin is sqrt(2).

    Usage:   [p_P_hom_norm, T] = normalise2dpts(pts)

    Argument:
      p_P_hom -  3xN array of 2D homogeneous coordinates

    Returns:
      p_P_hom_norm  -  3xN array of transformed 2D homogeneous coordinates.
      T             -  The 3x3 transformation matrix, pts_tilde = T*pts
    """
    p_P_non_canonical = p_P_hom / p_P_hom[2, :]
    p_P = p_P_hom[:2, :] / p_P_hom[2, :]

    # Centroid (Euclidean coordinates). Shape (2,): x and y coordinates
    mu = np.mean(p_P, axis=1)
    p_P_centered = (p_P.T - mu).T

    sigma = np.sqrt(np.mean(np.sum(p_P_centered**2, axis=0)))

    s = np.sqrt(2) / sigma
    T = np.array(
        [
            [s, 0, -s * mu[0]],
            [0, s, -s * mu[1]],
            [0, 0, 1],
        ]
    )

    p_P_hom_norm = T @ p_P_non_canonical
    # p_P_hom_norm = T @ p_P_hom  # TODO: gives slightly different results

    return p_P_hom_norm, T
