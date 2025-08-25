from src.mapping.structure_from_motion.fundamental_eight_point import (
    fundamentalEightPoint,
)
from src.mapping.structure_from_motion.normalise_2D_pts import normalise2DPts


def fundamentalEightPointNormalized(p1_I_hom, p2_I_hom):
    """Normalized Version of the 8 Point algorith

    Input: point correspondences
     - p1_I_hom np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
     - p2_I_hom np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

    Output:
     - F np.ndarray(3,3) : fundamental matrix
    """
    p1_I_hom_norm, T_1 = normalise2DPts(p1_I_hom)
    p2_I_hom_norm, T_2 = normalise2DPts(p2_I_hom)

    F_tilde = fundamentalEightPoint(p1_I_hom=p1_I_hom_norm, p2_I_hom=p2_I_hom_norm)

    F = T_2.T @ F_tilde @ T_1

    return F
