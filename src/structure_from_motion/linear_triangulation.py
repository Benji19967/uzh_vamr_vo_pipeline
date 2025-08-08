import numpy as np

from structure_from_motion.utils import cross2Matrix


def linear_triangulation(
    p1_I_hom: np.ndarray, p2_I_hom: np.ndarray, M1: np.ndarray, M2: np.ndarray
) -> np.ndarray:
    """Linear Triangulation

    Input:
     - p1_I_hom np.ndarray(3, N): homogeneous coordinates of points in image 1
     - p2_I_hom np.ndarray(3, N): homogeneous coordinates of points in image 2
     - M1 np.ndarray(3, 4): projection matrix corresponding to first image
     - M2 np.ndarray(3, 4): projection matrix corresponding to second image

    Output:
     - p_W_hom np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    N = p1_I_hom.shape[1]
    p_W_hom = np.zeros((4, N))

    for i, (p1i, p2i) in enumerate(zip(p1_I_hom.T, p2_I_hom.T)):
        p1ix = cross2Matrix(p1i)
        p2ix = cross2Matrix(p2i)

        a1 = p1ix @ M1
        a2 = p2ix @ M2
        A = np.r_[a1, a2]

        # vh.shape (4, 4)
        _, _, vh = np.linalg.svd(A)
        p_W_hom[:, i] = vh.T[:, -1]

        p_W_hom /= p_W_hom[3, :]

    return p_W_hom
