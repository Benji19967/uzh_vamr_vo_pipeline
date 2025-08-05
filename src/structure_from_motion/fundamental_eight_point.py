import numpy as np


def fundamentalEightPoint(p1_P_hom: np.ndarray, p2_P_hom: np.ndarray) -> np.ndarray:
    """The 8-point algorithm for the estimation of the fundamental matrix F

    The eight-point algorithm for the fundamental matrix with a posteriori
    enforcement of the singularity constraint (det(F)=0).
    Does not include data normalization.

    Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

    Input: point correspondences
     - p1_P_hom np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
     - p2_P_hom np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

    Output:
     - F np.ndarray(3,3) : fundamental matrix
    """
    N = p1_P_hom.shape[1]

    Q = np.zeros((N, 9))
    for i in range(N):
        a = np.kron(p1_P_hom[:, i], p2_P_hom[:, i])
        Q[i, :] = a

    _, _, vh = np.linalg.svd(Q)
    vec_F = vh.T[:, -1]
    F = vec_F.reshape((3, 3))

    u, s, vh = np.linalg.svd(F)
    s_new = np.zeros((3, 3))
    s_new[0][0] = s[0]
    s_new[1][1] = s[1]
    F = u @ s_new @ vh

    return F
