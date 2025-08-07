import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def normalize(v):
    return v / np.linalg.norm(v)


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
