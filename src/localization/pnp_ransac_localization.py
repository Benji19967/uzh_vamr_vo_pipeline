import cv2
import numpy as np

from src.exceptions import FailedLocalizationError
from src.localization.estimate_pose_dlt import estimate_pose_dlt
from src.structures.pose import Pose
from src.transformations.transformations import camera_to_pixel, world_to_camera

NUM_ITERATIONS = 2000
PIXEL_TOLERANCE = 10
NUM_SAMPLES = 6
MIN_INLIER_COUNT = 6


def pnp_ransac_localization_cv2(
    p_I_keypoints: np.ndarray,
    p_W_landmarks: np.ndarray,
    K: np.ndarray,
):
    """
    :param p_I_keypoints: (2, N) with p=(x, y)
    :param p_W_landmarks: (3, N)
    :param K: camera matrix intrinsics

    where N is the number of keypoints

    :returns:
        - T_C_W: np.ndarray(3, 4): Camera pose in world coordinates.
        - inlier_mask (N,): False (outlier) / True (inlier)
        - camera_position (3,): Camera position in world coordinates
    """
    N = p_I_keypoints.shape[1]
    dist_coeffs = np.zeros((4, 1))
    success, rvec, t_C_W, inliers = cv2.solvePnPRansac(  # type: ignore
        objectPoints=p_W_landmarks.T.reshape(-1, 1, 3),
        imagePoints=p_I_keypoints.T.reshape(-1, 1, 2),
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        reprojectionError=8.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,  # type: ignore
    )
    if not success:
        raise FailedLocalizationError("RANSAC failed localize camera pose")

    R_C_W, _ = cv2.Rodrigues(rvec)  # type: ignore

    # TODO: should this be -R_C_W.T @ t_C_W
    camera_position = -R_C_W @ t_C_W

    def inliers_to_mask(inliers: np.ndarray) -> np.ndarray:
        mask = np.zeros(N, dtype=bool)
        if inliers is not None:
            mask[inliers.flatten()] = True
        return mask

    return Pose(R_C_W, t_C_W), inliers_to_mask(inliers), camera_position.reshape(-1)


# TODO: yields a rather large reprojection error sometimes (>500 or >1000px)
# the cv2 equivalent consistently yields <10px for the same inputs
def pnp_ransac_localization(
    p_I_keypoints: np.ndarray,
    p_W_landmarks: np.ndarray,
    K: np.ndarray,
):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.

    :param p_I_keypoints: (2, N) with p=(x, y)
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
    N = p_I_keypoints.shape[1]
    if N < NUM_SAMPLES:
        raise ValueError(
            f"Num points {N} is smaller than num samples {NUM_SAMPLES} in RANSAC"
        )

    # Initialize RANSAC
    best_inlier_mask = np.zeros(N, dtype=np.bool_)

    # (row, col) to (u, v)
    # p_I_keypoints = np.flip(p_I_keypoints, axis=0)

    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0
    M_C_W_guess = None

    # RANSAC
    for _ in range(NUM_ITERATIONS):
        # Model from k samples (DLT or P3P)
        indices = np.random.choice(np.arange(N), size=NUM_SAMPLES, replace=False)
        p_I_keypoint_sample = p_I_keypoints[:, indices]
        p_W_landmark_sample = p_W_landmarks[:, indices]

        M_C_W_guess = estimate_pose_dlt(
            p_I=p_I_keypoint_sample, p_W=p_W_landmark_sample, K=K
        )
        R_C_W_guess = M_C_W_guess[:, :3]
        t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        p_W_hom_landmarks = np.r_[p_W_landmarks, np.ones((1, N))]
        T_C_W_guess = np.r_[M_C_W_guess, np.ones((1, 4))]
        p_C_landmarks = world_to_camera(p_W_hom=p_W_hom_landmarks, T_C_W=T_C_W_guess)
        p_I_landmarks = camera_to_pixel(p_C_landmarks, K)
        difference = p_I_keypoints - p_I_landmarks
        errors = (difference**2).sum(0)
        is_inlier = errors < PIXEL_TOLERANCE**2

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= MIN_INLIER_COUNT:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        num_iteration_history.append(NUM_ITERATIONS)
        max_num_inliers_history.append(max_num_inliers)

    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        # Statement suggests the guess is better than running DLT again

        # M_C_W = estimatePoseDLT(
        #     p_I_keypoints[:, best_inlier_mask],
        #     p_W_landmarks[:, best_inlier_mask],
        #     K,
        # )

        assert M_C_W_guess is not None
        R_C_W = M_C_W_guess[:, :3]
        t_C_W = M_C_W_guess[:, -1]

    return (
        R_C_W,
        t_C_W,
        best_inlier_mask,
        max_num_inliers_history,
        num_iteration_history,
    )
