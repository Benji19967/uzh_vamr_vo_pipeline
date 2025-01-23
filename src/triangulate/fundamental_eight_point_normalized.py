from src.triangulate.fundamental_eight_point import fundamentalEightPoint
from src.triangulate.normalise_2D_pts import normalise2DPts


def fundamentalEightPointNormalized(p1, p2):
    """Normalized Version of the 8 Point algorith
    Input: point correspondences
     - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
     - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

    Output:
     - F np.ndarray(3,3) : fundamental matrix
    """
    p1_normalised, T_1 = normalise2DPts(p1)
    p2_normalised, T_2 = normalise2DPts(p2)

    F_tilde = fundamentalEightPoint(p1=p1_normalised, p2=p2_normalised)

    F = T_2.T @ F_tilde @ T_1

    return F
