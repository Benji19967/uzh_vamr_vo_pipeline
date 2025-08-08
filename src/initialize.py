import numpy as np

from features.keypoints import get_keypoint_correspondences
from image import Image
from localization import ransacLocalization
from structure_from_motion import sfm

MAX_NUM_KEYPOINTS = 1000


def initialize(
    I_0: Image,
    I_1: Image,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From two images find a set of corresponding 2D keypoints and compute the
    associated 3D landmarks

    Args:
     - I_0, I_1: images to extract corresponding keypoints from
     - K: camera matrix

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (
                (2xN) keypoints of image 1 in pixel coordinates (x, y),
                (2xN) keypoints of image 2 in pixel coordinates (x, y),
                (3xN) landmarks P_W in 3D coordinates (x, y, z)
            )
    """
    p1_P_keypoints, p2_P_keypoints = get_keypoint_correspondences(
        I_0=I_0, I_1=I_1, max_num_keypoints=MAX_NUM_KEYPOINTS
    )

    p_W, _, _ = sfm.run_sfm(p1_P=p1_P_keypoints, p2_P=p2_P_keypoints, K=K)
    _, _, best_inlier_mask, _, _ = ransacLocalization(
        p_P_keypoints=p1_P_keypoints,
        p_W_landmarks=p_W,
        K=K,
    )

    return (
        p1_P_keypoints[:, best_inlier_mask],
        p2_P_keypoints[:, best_inlier_mask],
        p_W[:, best_inlier_mask],
    )
