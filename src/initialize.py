import numpy as np

from src.features.keypoints import get_keypoint_correspondences
from src.image import Image
from src.localization.localization import ransacLocalization
from src.structure_from_motion import sfm

MAX_NUM_KEYPOINTS = 1000


def initialize(
    image_0: Image,
    image_1: Image,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From two images find a set of corresponding 2D keypoints and compute the
    associated 3D landmarks.

    Run RANSAC to remove outliers.

    Args:
     - image_0, image_1: images to extract corresponding keypoints from
     - K: camera matrix

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (
                (2xN) keypoints of image 1 in pixel coordinates (x, y),
                (2xN) keypoints of image 2 in pixel coordinates (x, y),
                (3xN) landmarks P_W in 3D coordinates (x, y, z)
            )
    """
    p1_I_keypoints, p2_I_keypoints = get_keypoint_correspondences(
        image_0=image_0, image_1=image_1, max_num_keypoints=MAX_NUM_KEYPOINTS
    )

    p_W, _, _ = sfm.run_sfm(p1_I=p1_I_keypoints, p2_I=p2_I_keypoints, K=K)
    _, _, best_inlier_mask, _, _ = ransacLocalization(
        p_I_keypoints=p1_I_keypoints,
        p_W_landmarks=p_W,
        K=K,
    )

    return (
        p1_I_keypoints[:, best_inlier_mask],
        p2_I_keypoints[:, best_inlier_mask],
        p_W[:, best_inlier_mask],
    )
