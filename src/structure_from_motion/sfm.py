import numpy as np

from structure_from_motion.decompose_essential_matrix import decomposeEssentialMatrix
from structure_from_motion.disambiguate_relative_pose import disambiguateRelativePose
from structure_from_motion.estimate_essential_matrix import estimateEssentialMatrix
from structure_from_motion.linear_triangulation import linear_triangulation


def run_sfm(p1_I: np.ndarray, p2_I: np.ndarray, K: np.ndarray):
    """Run structure from motion

    Input:
     - p1_I np.ndarray(2, N): coordinates of points in image 1
     - p2_I np.ndarray(2, N): coordinates of points in image 2
     - K    np.ndarray(3, 3): calibration matrix of camera 1

    Output:
     - p_W_hom              np.ndarray(4, N): homogeneous coordinates of 3-D points
     - camera_position_W    np.ndarray(3, 1): position of camera
     - camera_direction_W   np.ndarray(3, 3): rotation of camera
    """
    N = p1_I.shape[1]
    p1_I_hom = np.r_[p1_I, np.ones((1, N))]
    p2_I_hom = np.r_[p2_I, np.ones((1, N))]

    # Estimate the essential matrix E using the 8-point algorithm
    E = estimateEssentialMatrix(p1_I_hom=p1_I_hom, p2_I_hom=p2_I_hom, K1=K, K2=K)

    # Extract the relative camera positions (R,T) from the essential matrix
    # Obtain extrinsic parameters (R,t) from E
    Rots, u3 = decomposeEssentialMatrix(E)

    # Disambiguate among the four possible configurations
    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1_I_hom, p2_I_hom, K, K)

    # Triangulate a point cloud using the final transformation (R,T)
    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]

    p_W_hom = linear_triangulation(p1_I_hom=p1_I_hom, p2_I_hom=p2_I_hom, M1=M1, M2=M2)
    p_W = p_W_hom[:3, :]

    # TODO: why transpose? Just to fit `draw_camera()`?
    camera_position_W = -R_C2_W.T @ T_C2_W
    camera_direction_W = R_C2_W.T

    return p_W, camera_position_W, camera_direction_W
