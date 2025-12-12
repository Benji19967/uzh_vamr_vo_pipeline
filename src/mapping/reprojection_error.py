import cv2
import numpy as np

from src.structures.keypoints2D import Keypoints2D
from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose


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
    errors = reprojection_errors(p_W_hom, p_I, T_C_W, K)
    return np.mean(errors)


def reprojection_errors(
    p_W_hom: np.ndarray,
    p_I: np.ndarray,
    T_C_W: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Computes reprojection errors for a set of N points.

    Args:
        p_W_hom (np.ndarray): shape (4, N), homogeneous 3D points.
        p_I (np.ndarray): shape (2, N), 2D image points.
        T_C_W (np.ndarray): shape (3, 4), camera pose.
        K (np.ndarray): shape (3, 3), camera intrinsics.

    Returns:
        np.ndarray(N,): reprojection error for each 2D/3D point pair
    """
    M = K @ T_C_W

    # Project 3D points to image
    p_proj_hom = M @ p_W_hom
    p_proj = p_proj_hom[:2] / p_proj_hom[2]  # Normalize

    # Compute pixel-wise error
    errors = np.linalg.norm(p_proj - p_I, axis=0)  # shape (N,)
    return errors


def reprojection_errors_ba(
    x,
    all_poses: list[Pose],
    all_landmarks: Landmarks3D,
    observations: list,  # list of (cam_id, point_id, u_obs, v_obs)
    K: np.ndarray,
    # num_cams: int,
    # num_points: int,
):
    """
    Returns:
        np.ndarray(2*num_observations,): reprojection error for each observation
    """
    x = np.asarray(x)

    # cam_block = 6 * num_cams
    # rts = x[:cam_block].reshape(num_cams, 6)
    # points_3d = x[cam_block:].reshape(num_points, 3)
    #
    # rvecs = rts[:, :3]
    # tvecs = rts[:, 3:]

    errors = []

    for cam_id, point_id, u_obs, v_obs in observations:

        # get parameters for this camera and point
        pose = all_poses[cam_id]
        # rvec = rvecs[cam_id]
        # tvec = tvecs[cam_id]
        p_W = all_landmarks.array[:, point_id].reshape(1, 1, 3)

        rvec = pose.rvec.ravel().astype(np.float64)  # shape (3,)
        tvec = pose.tvec.ravel().astype(np.float64)  # shape (3,)

        # project using OpenCV (Rodrigues inside)
        uv_proj, _ = cv2.projectPoints(p_W, rvec, tvec, K, None)  # type: ignore

        u_proj, v_proj = uv_proj.ravel()

        # reprojection error
        errors.append(u_proj - u_obs)
        errors.append(v_proj - v_obs)

    return np.array(errors)
