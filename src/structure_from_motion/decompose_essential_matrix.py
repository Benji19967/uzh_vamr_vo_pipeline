import numpy as np


def decomposeEssentialMatrix(E):
    """Given an essential matrix, compute the camera motion, i.e.,  R and T such
    that E ~ T_x R

    Input:
      - E(3,3) : Essential matrix

    Output:
      - R(3,3,2) : the two possible rotations
      - u3(3,1)   : a vector with the translation information
    """
    pass
    u, s, vh = np.linalg.svd(E)
    u3 = u[:, -1]
    W = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    R = np.zeros((3, 3, 2))
    r1 = u @ W @ vh
    r2 = u @ W.T @ vh
    for r in [r1, r2]:
        if np.linalg.det(r) < 0:
            r *= -1
    R[:, :, 0] = r1
    R[:, :, 1] = r2

    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    return R, u3
