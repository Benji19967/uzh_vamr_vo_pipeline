import numpy as np


def reprojection_error(
    p_W_hom: np.ndarray,
    p_I: np.ndarray,
    T_C_W: np.ndarray,
    K: np.ndarray,
) -> float:
    """
    Computes mean reprojection error for a set of N points.

    Args:
        p_W_hom (np.ndarray): shape (4, N), homogeneous 3D points.
        p_I (np.ndarray): shape (2, N), 2D image points.
        T_C_W (np.ndarray): shape (3, 4), camera pose.
        K (np.ndarray): shape (3, 3), camera intrinsics.

    Returns:
        float: Mean reprojection error.
    """
    M = K @ T_C_W

    # Project 3D points to image
    p_proj_hom = M @ p_W_hom
    p_proj = p_proj_hom[:2] / p_proj_hom[2]  # Normalize

    # Compute pixel-wise error
    errors = np.linalg.norm(p_proj - p_I, axis=0)  # shape (N,)
    return np.mean(errors)
