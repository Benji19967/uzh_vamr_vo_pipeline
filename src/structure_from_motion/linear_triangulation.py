import numpy as np

from src.structure_from_motion.utils import cross2Matrix


def reprojection_error(
    p_W_hom: np.ndarray,  # shape (4, N): homogeneous 3D points
    p_I: np.ndarray,  # shape (2, N): 2D image points
    T_C_W: np.ndarray,  # shape (3, 4): camera pose
    K: np.ndarray,  # shape (3, 3): camera intrinsics
) -> float:
    """
    Computes mean reprojection error for a set of points.

    Returns:
        Mean reprojection error (float)
    """
    N = p_W_hom.shape[1]
    P = K @ T_C_W  # Projection matrix: shape (3, 4)

    # Project 3D points to image
    p_proj_hom = P @ p_W_hom  # shape (3, N)
    p_proj = p_proj_hom[:2] / p_proj_hom[2]  # Normalize

    # Compute pixel-wise error
    errors = np.linalg.norm(p_proj - p_I, axis=0)  # shape (N,)
    print("ERRORS")
    print(errors)
    return np.mean(errors)


# def reprojection_error(
#     p_W_hom: np.ndarray, p_I: np.ndarray, T_C_W: np.ndarray, K: np.ndarray
# ) -> float:
#     """
#     Input:
#      - p_W_hom np.ndarray(4, 1): homogeneous coordinates 3D point
#      - p_I np.ndarray(2, 1): coordinates of point in image
#      - T_C_W np.ndarray(3, 4): transformation matrix
#      - K np.ndarray(3, 3): camera matrix
#
#      Output:
#       - reprojection error
#     """
#     M = K @ T_C_W
#     p_proj_hom = M @ p_W_hom
#     p_proj = p_proj_hom[:2] / p_proj_hom[2]
#     return np.linalg.norm(p_proj - p_I)


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
