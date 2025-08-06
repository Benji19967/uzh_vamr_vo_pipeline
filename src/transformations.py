import numpy as np

from distort_points import distort_points


def world_to_camera(
    p_W_hom: np.ndarray,
    T_C_W: np.ndarray,
) -> np.ndarray:
    """
    Transformation from World to Camera frame

    Args:
        p_W_hom: 3d points in World frame and as homogeneous coordinates (4xN)
        T_C_W: 4x4 transformation matrix to map points from world to camera frame

    Returns:
        p_C: 3d points in camera frame (3xN)
    """
    p_C = np.matmul(T_C_W[:3, :], p_W_hom)

    return p_C


# def world_to_camera_non_hom(
#     p_W: np.ndarray,
#     R_C_W: np.ndarray,
#     t_C_W: np.ndarray,
# ) -> np.ndarray:
#     """
#     Transformation from World to Camera frame

#     Args:
#         p_W: (3xN) 3d points in World frame
#         R_C_W: 3x3 rotation matrix to map points from world to camera frame
#         t_C_W: 3x1 translation vector to map points from world to camera frame

#     Returns:
#         p_C: 3d points in camera frame (3xN)
#     """
#     print(p_W)
#     print(p_W.shape)
#     print(p_W[:, :, None])
#     print(p_W[:, :, None].shape)
#     p_C = np.matmul(R_C_W, p_W) + t_C_W

#     return p_C


def camera_to_pixel(
    p_C: np.ndarray, K: np.ndarray, D: np.ndarray | None = None
) -> np.ndarray:
    """
    Projects 3d points from the camera frame to the image plane, given the camera matrix.

    If distortion coefficients as provided, apply distortion.

    Args:
        p_C: 3d points in camera frame (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        p_P: 2d points in pixel coordinates (2xN)
    """
    u_v_lambda = np.matmul(K, p_C)
    p_P = u_v_lambda[:2, :] / u_v_lambda[2]

    if D is not None:
        p_P_distorted = distort_points(x=p_P, K=K, D=D)
        return p_P_distorted

    return p_P
