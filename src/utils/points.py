import numpy as np


def apply_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to points if points are not empty.

    Args:
        - points  np.ndarray(Any, N)
        - mask    np.ndarray(N,)
    """

    if points.any():
        return points[:, mask]
    return points


def apply_mask_many(points: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    """
    Apply mask to list of points if points are not empty.

    Args:
        - points  list[np.ndarray(Any, N)]
        - mask    np.ndarray(N,)
    """
    masked = []
    for pts in points:
        masked.append(apply_mask(pts, mask))
    return masked


def to_cv2(p_I: np.ndarray) -> np.ndarray:
    """
    Convert format of points to match what cv2 expects

    Args:
        p_I (np.ndarray): (2xN)

    Returns:
        np.ndarray: (Nx1x2)
    """
    p_I = p_I.astype(np.float32)
    N = p_I.shape[1]
    return p_I.T.reshape((N, 1, 2))


def from_cv2(p_I: np.ndarray) -> np.ndarray:
    """
    Convert format of points from cv2 format to (2xN)

    Args:
        p_I (np.ndarray): (Nx1x2)

    Returns:
        np.ndarray: (2xN)
    """
    return p_I.T.reshape(2, -1)


def compute_bearing_angles_with_translation(
    p_I_1: np.ndarray,
    p_I_2: np.ndarray,
    poses_A: np.ndarray,
    T_C_W: np.ndarray,
    K: np.ndarray,
    min_angle: float = 5.0,
):
    """
    Compute angles between bearing vectors (with translation) and return a mask of angles > min_angle degrees.

    Inputs:
        p_I_1, p_I_2: np.ndarray(2, N) arrays of 2D image points
        poses_A: np.ndarray(12, N) arrays of flattened 3x4 pose matrices, one per correspondence
        T_C_W:  flattened 3x4 pose matrix
        K: np.ndarray(3, 3) camera intrinsic matrix

    Returns:
        angles_rad: (N,) array of angles in radians
        angles_deg: (N,) array of angles in degrees
        angle_mask: (N,) boolean array, True if angle > 5 degrees
    """
    N = p_I_1.shape[1]

    # Convert points to homogeneous coords
    ones = np.ones((1, N))
    pts_A_h = np.vstack((p_I_1, ones))  # (3, N)
    pts_B_h = np.vstack((p_I_2, ones))  # (3, N)

    K_inv = np.linalg.inv(K)

    # Backproject to normalized camera coords
    dirs_A_cam = K_inv @ pts_A_h  # (3, N)
    dirs_B_cam = K_inv @ pts_B_h  # (3, N)

    # Reshape poses_A and pose_B
    poses_A_reshaped = poses_A.reshape(3, 4, N)  # (3,4,N)
    R_A = poses_A_reshaped[:, :3, :]  # (3,3,N)
    t_A = poses_A_reshaped[:, 3, :]  # (3,N)

    pose_B_reshaped = T_C_W.reshape(3, 4)  # (3,4)
    R_B = pose_B_reshaped[:, :3]  # (3,3)
    t_B = pose_B_reshaped[:, 3]  # (3,)

    # Camera centers in world frame
    # C_A = np.empty((3, N))
    # for i in range(N):
    #     C_A[:, i] = -R_A[:, :, i].T @ t_A[:, i]
    # C_B = -R_B.T @ t_B

    # Rotate directions to world frame
    dirs_A_world = np.empty_like(dirs_A_cam)
    for i in range(N):
        dirs_A_world[:, i] = R_A[:, :, i].T @ dirs_A_cam[:, i]
    dirs_B_world = R_B.T @ dirs_B_cam  # same for all points

    # Normalize direction vectors
    dirs_A_world /= np.linalg.norm(dirs_A_world, axis=0, keepdims=True)
    dirs_B_world /= np.linalg.norm(dirs_B_world, axis=0, keepdims=True)

    # Now compute angle between the two bearing vectors for each point
    dots = np.sum(dirs_A_world * dirs_B_world, axis=0)
    dots = np.clip(dots, -1.0, 1.0)

    angles_rad = np.arccos(dots)
    angles_deg = np.degrees(angles_rad)

    # Mask for angles above min_angle degrees
    angle_mask = angles_deg > min_angle

    return angles_rad, angles_deg, angle_mask
