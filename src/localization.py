import numpy as np

from estimate_pose_dlt import estimatePoseDLT
from transformations import camera_to_pixel, world_to_camera

NUM_ITERATIONS = 2000
PIXEL_TOLERANCE = 10
NUM_SAMPLES = 6


def ransacLocalization(
    p_P_keypoints: np.ndarray,
    p_W_landmarks: np.ndarray,
    K: np.ndarray,
):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.

    :param p_P_keypoints: (2, N) with p=(x, y)
    :param p_W_landmarks: (3, N)
    :param K: camera matrix intrinsics

    where N is the number of keypoints

    :returns:
        - R_C_W
        - t_C_W
        - best_inlier_mask: (1, num_matched) False (outlier) / True (inlier)
        - max_num_inliers_history
        - num_iteration_history
    """
    N = p_P_keypoints.shape[1]
    # Initialize RANSAC
    best_inlier_mask = np.zeros(N)

    # (row, col) to (u, v)
    # p_P_keypoints = np.flip(p_P_keypoints, axis=0)

    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    for _ in range(NUM_ITERATIONS):
        # Model from k samples (DLT or P3P)
        indices = np.random.choice(np.arange(N), size=NUM_SAMPLES, replace=False)
        p_P_keypoint_sample = p_P_keypoints[:, indices]
        p_W_landmark_sample = p_W_landmarks[:, indices]

        M_C_W_guess = estimatePoseDLT(
            p_P=p_P_keypoint_sample, p_W=p_W_landmark_sample, K=K
        )
        R_C_W_guess = M_C_W_guess[:, :3]
        t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        p_W_hom_landmarks = np.r_[p_W_landmarks, np.ones((1, N))]
        T_C_W_guess = np.r_[M_C_W_guess, np.ones((1, 4))]
        p_C_landmarks = world_to_camera(p_W_hom=p_W_hom_landmarks, T_C_W=T_C_W_guess)
        p_P_landmarks = camera_to_pixel(p_C_landmarks, K)
        difference = p_P_keypoints - p_P_landmarks
        errors = (difference**2).sum(0)
        is_inlier = errors < PIXEL_TOLERANCE**2

        min_inlier_count = 6

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        num_iteration_history.append(NUM_ITERATIONS)
        max_num_inliers_history.append(max_num_inliers)

    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W = estimatePoseDLT(
            p_P_keypoints[:, best_inlier_mask],
            p_W_landmarks[:, best_inlier_mask],
            K,
        )
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

    return (
        R_C_W,
        t_C_W,
        best_inlier_mask,
        max_num_inliers_history,
        num_iteration_history,
    )
