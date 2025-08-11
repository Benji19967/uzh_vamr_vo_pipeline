import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def normalize(v):
    return v / np.linalg.norm(v)


MIN_ANGLE = 10.0


def compute_bearing_angles_with_translation(
    p_I_1: np.ndarray,
    p_I_2: np.ndarray,
    poses_A: np.ndarray,
    T_C_W: np.ndarray,
    K: np.ndarray,
):
    """
    Compute angles between bearing vectors (with translation) and return a mask of angles > 5 degrees.

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

    # Mask for angles above MIN_ANGLE degrees
    angle_mask = angles_deg > MIN_ANGLE

    return angles_rad, angles_deg, angle_mask


def plot_angle(x1, x2, K, R1, t1, R2, t2):
    # Camera intrinsics (identity matrix for visualization simplicity)
    # K = np.eye(3)

    # Matched keypoints in image 1 and image 2
    # x1 = np.array([100, 120])
    # x2 = np.array([130, 115])

    # Camera poses
    # R1 = np.eye(3)
    # t1 = np.array([0, 0, 0])  # Camera 1 at origin

    # R2 = np.eye(3)
    # t2 = np.array([1, 0, 0])  # Camera 2 translated 1 unit to the right

    # Convert to homogeneous coordinates
    x1_h = np.append(x1, 1)
    x2_h = np.append(x2, 1)

    # Step 1: Get bearing vectors in camera frame
    b1_c = normalize(np.linalg.inv(K) @ x1_h)
    b2_c = normalize(np.linalg.inv(K) @ x2_h)

    # Step 2: Rotate bearing vectors to world frame
    b1_w = normalize(R1.T @ b1_c)
    b2_w = normalize(R2.T @ b2_c)

    # Step 3: Compute camera centers
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    # Step 4: Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Camera centers
    ax.scatter(*C1, color="blue", label="Camera 1")
    ax.scatter(*C2, color="green", label="Camera 2")

    # Bearing vectors (rays)
    scale = 3
    ax.quiver(*C1, *(b1_w * scale), color="blue", arrow_length_ratio=0.1)
    ax.quiver(*C2, *(b2_w * scale), color="green", arrow_length_ratio=0.1)

    # Labels
    ax.text(*C1, "Cam1", color="blue")
    ax.text(*C2, "Cam2", color="green")

    # Compute angle between the two world-frame rays
    dot = np.dot(b1_w, b2_w)
    angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    # Plot formatting
    ax.set_title(f"Angle Between Bearing Vectors: {angle_deg:.2f}°")
    ax.set_xlim([-1, 5])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return angle_deg
