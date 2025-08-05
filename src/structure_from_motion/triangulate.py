import numpy as np
from decompose_essential_matrix import decomposeEssentialMatrix
from disambiguate_relative_pose import disambiguateRelativePose
from estimate_essential_matrix import estimateEssentialMatrix
from linear_triangulation import linear_triangulation


def triangulate(p1_P: np.ndarray, p2_P: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Structure from motion

    Input:
     - p1_P np.ndarray(2, N): coordinates of points in image 1
     - p2_P np.ndarray(2, N): coordinates of points in image 2
     - K    np.ndarray(3, 3): calibration matrix of camera 1

    Output:
     - p_W_hom np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    N = p1_P.shape[1]
    p1_P_hom = np.r_[p1_P, np.ones((1, N))]
    p2_P_hom = np.r_[p2_P, np.ones((1, N))]

    # Estimate the essential matrix E using the 8-point algorithm
    E = estimateEssentialMatrix(p1_P_hom=p1_P_hom, p2_P_hom=p2_P_hom, K1=K, K2=K)

    # Extract the relative camera positions (R,T) from the essential matrix
    # Obtain extrinsic parameters (R,t) from E
    Rots, u3 = decomposeEssentialMatrix(E)

    # Disambiguate among the four possible configurations
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1_P_hom, p2_P_hom, K, K)

    # Triangulate a point cloud using the final transformation (R,T)
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]

    p_W_hom = linear_triangulation(p1_P_hom=p1_P_hom, p2_P_hom=p2_P_hom, M1=M1, M2=M2)
    p_W = p_W_hom[:3, :]

    return p_W
