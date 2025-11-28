import logging
from pathlib import Path

import numpy as np

from src.exceptions import FailedLocalizationError
from src.features import keypoints
from src.localization.pnp_ransac_localization import pnp_ransac_localization_cv2
from src.mapping.reprojection_error import reprojection_error
from src.mapping.triangulate_landmarks import triangulate_landmarks
from src.plotting.visualizer import Visualizer
from src.tracking.klt import run_klt
from src.utils import points
from src.utils.masks import compose_masks
from src.utils.points import compute_bearing_angles_with_translation

np.set_printoptions(suppress=True)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HERE = Path(__file__).parent
BA_DATA_FILENAME = HERE / ".." / "ba_data" / "ba_data.txt"

MAX_NUM_CANDIDATE_KEYPOINTS = 1000
MAX_NUM_NEW_CANDIDATE_KEYPOINTS = 1000
MAX_REPROJECTION_ERROR = 5  # pixels
MIN_ANGLE_TO_TRIANGULATE = 5.0  # degrees
MIN_NUM_LANDMARKS_FOR_LOCALIZATION = 4
KEYFRAME_INTERVAL = 5  # Process every ith image as a keyframe


def run_vo(
    images: list[np.ndarray],
    p_I_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
    K: np.ndarray,
    plot_keypoints: bool,
    plot_landmarks: bool,
    plot_tracking: bool,
    plot_reprojection_errors: bool,
    plot_scale_drift: bool,
    plot_trajectory: bool,
    camera_positions_ground_truth: list[np.ndarray] | None = None,
):
    """
    Run a visual odometry pipeline on the images

    Args:
        - images list[np.ndarray]
        - p_I_keypoints_initial np.ndarray(2,N)   | (x,y)
        - p_W_landmarks_initial: np.ndarray(3, N) | (x,y,z)
        - K np.ndarray(3, 3): camera matrix
    """
    with open(BA_DATA_FILENAME, "w") as f:
        f.write(f"{len(images) / KEYFRAME_INTERVAL}\n")

    visualizer = Visualizer(
        plot_keypoints,
        plot_landmarks,
        plot_tracking,
        plot_reprojection_errors,
        plot_scale_drift,
        plot_trajectory,
    )
    P1, X1, C1, F1, T1 = initialize_state(p_I_keypoints_initial, p_W_landmarks_initial)

    camera_positions = []
    reprojection_errors = []
    for i, (img_0, img_1) in enumerate(zip(images, images[1:])):
        logger.debug(f"Iteration: {i}")

        # Track keypoints from img_0 to img_1
        P1, status_mask = run_klt(img_0, img_1, P1)
        P1, X1 = points.apply_mask_many([P1, X1], status_mask)

        # Track candidate keypoints from img_0 to img_1
        C0 = C1
        C1, status_mask_candiate_kps = run_klt(img_0, img_1, C1)
        C0, C1, F1, T1 = points.apply_mask_many(
            [C0, C1, F1, T1], status_mask_candiate_kps
        )
        visualizer.tracking(C0, C1, img_0)
        logger.debug(f"After klt: P1: {P1.shape}, X1: {X1.shape}, C1: {C1.shape}")

        # Localize: compute camera pose
        if P1.shape[1] < MIN_NUM_LANDMARKS_FOR_LOCALIZATION:
            raise ValueError(f"Not enough keypoints/landmarks for localization")
        try:
            T_C_W, best_inlier_mask, camera_position = pnp_ransac_localization_cv2(
                P1, X1, K
            )
        except FailedLocalizationError:
            logger.debug(f"Failed Ransac localization")
            continue
        P1, X1 = points.apply_mask_many([P1, X1], best_inlier_mask)
        camera_positions.append(camera_position)
        logger.debug(f"After ran: P1: {P1.shape}, X1: {X1.shape}, C1: {C1.shape}")
        logger.debug(f"Pose:\n {T_C_W}")
        logger.debug(f"Camera position: {camera_position.flatten()}")

        # Map: add new landmarks
        if i % KEYFRAME_INTERVAL == 0:
            C1, F1, T1 = add_new_candidate_keypoints(img_1, P1, C1, F1, T1, T_C_W)
            P1, X1, C1, F1, T1 = add_new_landmarks(P1, X1, C1, F1, T1, T_C_W, K)
            with open(BA_DATA_FILENAME, "a+") as f:
                f.write(f"{X1.shape[1]}\n")
                np.savetxt(f, T_C_W.flatten())
                np.savetxt(f, P1.T)
                np.savetxt(f, X1.T)
        if C1.shape[1] > MAX_NUM_CANDIDATE_KEYPOINTS:
            C1 = C1[:, -MAX_NUM_CANDIDATE_KEYPOINTS:]
            F1 = F1[:, -MAX_NUM_CANDIDATE_KEYPOINTS:]
            T1 = T1[:, -MAX_NUM_CANDIDATE_KEYPOINTS:]

        # Evaluate results
        reproj_error = reprojection_error(points.to_hom(X1), P1, T_C_W, K)
        reprojection_errors.append(reproj_error)
        logger.debug(f"Reprojection error landmarks: {reproj_error}")

        visualizer.keypoints_and_landmarks(P1, X1, C1, camera_positions, img_1)
    visualizer.trajectory(camera_positions, camera_positions_ground_truth)
    visualizer.reprojection_errors(reprojection_errors)
    if plot_scale_drift:
        assert camera_positions_ground_truth
        visualizer.scale_drift(camera_positions, camera_positions_ground_truth)


def add_new_candidate_keypoints(img_1: np.ndarray, P1, C1, F1, T1, T_C_W):
    """
    Add new candidate keypoints to the current set of keypoints.

    Args:
        img_1: np.ndarray: Current image.
        P1: np.ndarray(2, N): Current keypoints.
        C1: np.ndarray(2, M): Current candidate keypoints.
        F1: np.ndarray(2, M): First track of current candidate keypoints.
        T1: np.ndarray(12, M): Camera poses at first track of current candidate keypoints.
        T_C_W: np.ndarray(3, 4): Camera pose for the current image.
    """
    C1_new, num_new_candidate_keypoints = keypoints.find_keypoints(
        img_1, MAX_NUM_NEW_CANDIDATE_KEYPOINTS, exclude=[C1, P1]
    )
    C1 = np.c_[C1, C1_new]
    F1 = np.c_[F1, C1_new]
    T1 = np.c_[T1, multiply_T(get_T_C_W_flat(T_C_W), num_new_candidate_keypoints)]
    return C1, F1, T1


def add_new_landmarks(P1, X1, C1, F1, T1, T_C_W, K):
    """
    Add new landmarks to the current set of landmarks.
    Remove the corresponding candidate keypoints from the current set.

    Requirements to find a new landmark:

    1. Bearing angle > threshold
    2. Reprojection error < threshold
    3. Is not an outlier when using RANSAC

    Args:
        P1: np.ndarray(2, N): Current keypoints.
        X1: np.ndarray(3, N): Current landmarks.
        C1: np.ndarray(2, M): Current candidate keypoints.
        F1: np.ndarray(2, M): First track of current candidate keypoints.
        T1: np.ndarray(12, M): Camera poses at first track of current candidate keypoints.
        T_C_W: np.ndarray(3, 4): Camera pose for the current image.
        K: np.ndarray(3, 3): Camera intrinsic matrix.
    Returns:
        P1: np.ndarray(2, N): Updated keypoints.
        X1: np.ndarray(3, N): Updated landmarks.
        C1: np.ndarray(2, M): Updated candidate keypoints.
    """
    assert F1.any()
    _, _, mask_to_triangulate = compute_bearing_angles_with_translation(
        F1, C1, T1, T_C_W, K, MIN_ANGLE_TO_TRIANGULATE
    )

    p_W_new_landmarks, mask_successful_triangulation = triangulate_landmarks(
        F1, C1, T1, T_C_W, K, mask_to_triangulate, MAX_REPROJECTION_ERROR
    )
    C1_triangulated = points.apply_mask(C1, mask_successful_triangulation)
    logger.debug(f"Successful triangulation: {mask_successful_triangulation.sum()}")

    best_inlier_mask_ransac = np.full(mask_successful_triangulation.sum(), False)
    if C1.any() and mask_successful_triangulation.sum() >= 4:
        _, best_inlier_mask_ransac, _ = pnp_ransac_localization_cv2(
            C1_triangulated, p_W_new_landmarks, K
        )
        logger.debug(f"Num ransac inliers: {best_inlier_mask_ransac.sum()}")

        C1_triangulated_inliers, p_W_new_landmarks_inliers = points.apply_mask_many(
            [C1_triangulated, p_W_new_landmarks],
            best_inlier_mask_ransac,
        )

        mask_new_landmarks = compose_masks(
            mask_successful_triangulation, best_inlier_mask_ransac
        )

        P1 = np.c_[P1, C1_triangulated_inliers]
        X1 = np.c_[X1, p_W_new_landmarks_inliers]
        C1, F1, T1 = points.apply_mask_many([C1, F1, T1], ~mask_new_landmarks)

    return P1, X1, C1, F1, T1


def multiply_T(T_C_W_flat, num_new_candidate_keypoints):
    """
    From (1, 12):

    [x, y, z, ..., v]

    to (12, num_new_candidate_keypoints)

    [
        [x, x, x, ...],
        [y, y, y, ...],
        [z, z, z, ...],
        ...
        [v, v, v, ...]
    ]

    """
    return np.tile(T_C_W_flat, (num_new_candidate_keypoints, 1)).T


def initialize_state(
    p_I_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
):
    """
    S1 = (P1,X1,C1,F1,T1)

    P1: np.ndarray(2, N): Current keypoints.
    X1: np.ndarray(3, N): Current landmarks.
    C1: np.ndarray(2, M): Current candidate keypoints.
    F1: np.ndarray(2, M): First track of current candidate keypoints.
    T1: np.ndarray(12, M): Camera poses at first track of current candidate keypoints.
    """
    P1 = p_I_keypoints_initial
    X1 = p_W_landmarks_initial
    C1 = np.zeros((2, 0), dtype=np.int32)
    F1 = np.zeros((2, 0), dtype=np.int32)
    T1 = np.zeros((12, 0), dtype=np.int32)

    return P1, X1, C1, F1, T1


def get_T_C_W_flat(T_C_W):
    """
    From (3, 4):

    r11 r12 r13 tx
    r21 r22 r23 ty
    r31 r32 r33 tz

    to (1, 12):

    r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    """
    return T_C_W.flatten()
