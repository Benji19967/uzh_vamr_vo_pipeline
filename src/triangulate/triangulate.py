import numpy as np


def cross2Matrix(x):
    """Antisymmetric matrix corresponding to a 3-vector
    Computes the antisymmetric matrix M corresponding to a 3-vector x such
    that M*y = cross(x,y) for all 3-vectors y.

    Input:
      - x np.ndarray(3,1) : vector

    Output:
      - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return M


def linear_triangulation(
    p1_P_hom: np.ndarray, p2_P_hom: np.ndarray, M1: np.ndarray, M2: np.ndarray
) -> np.ndarray:
    """Linear Triangul∂∂ation

    Input:
     - p1_P_hom np.ndarray(3, N): homogeneous coordinates of points in image 1
     - p2_P_hom np.ndarray(3, N): homogeneous coordinates of points in image 2
     - M1 np.ndarray(3, 4): projection matrix corresponding to first image
     - M2 np.ndarray(3, 4): projection matrix corresponding to second image

    Output:
     - p_W_hom np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    N = p1_P_hom.shape[1]
    p_W_hom = np.zeros((4, N))

    for i, (p1i, p2i) in enumerate(zip(p1_P_hom.T, p2_P_hom.T)):
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
