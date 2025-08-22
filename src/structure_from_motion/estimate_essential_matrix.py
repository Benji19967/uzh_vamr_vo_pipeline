from src.structure_from_motion.fundamental_eight_point_normalized import (
    fundamentalEightPointNormalized,
)


def estimateEssentialMatrix(p1_I_hom, p2_I_hom, K1, K2):
    """estimates the essential matrix given matching point coordinates,
       and the camera calibration K

    Input: point correspondences
     - p1_I_hom   np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
     - p2_I_hom   np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2
     - K1         np.ndarray(3,3): calibration matrix of camera 1
     - K2         np.ndarray(3,3): calibration matrix of camera 2

    Output:
     - E np.ndarray(3,3) : essential matrix
    """
    F = fundamentalEightPointNormalized(p1_I_hom=p1_I_hom, p2_I_hom=p2_I_hom)

    E = K2.T @ F @ K1

    return E
