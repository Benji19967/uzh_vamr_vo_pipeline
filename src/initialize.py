import numpy as np

from src.features.keypoints import get_keypoint_correspondences
from src.localization.pnp_ransac_localization import pnp_ransac_localization_cv2
from src.mapping.structure_from_motion import sfm
from src.structures.keypoints2D import Keypoints2D
from src.structures.landmarks3D import Landmarks3D
from src.utils.image import Image

MAX_NUM_KEYPOINTS = 1000


def initialize(
    image_0: Image,
    image_1: Image,
    K: np.ndarray,
) -> tuple[Keypoints2D, Keypoints2D, Landmarks3D]:
    """
    From two images find a set of corresponding 2D keypoints and compute the
    associated 3D landmarks.

    Run RANSAC to remove outliers.

    Args:
     - image_0, image_1: images to extract corresponding keypoints from
     - K: camera matrix
    """
    p1_I_keypoints, p2_I_keypoints = get_keypoint_correspondences(
        image_0=image_0, image_1=image_1, max_num_keypoints=MAX_NUM_KEYPOINTS
    )

    p_W, _, _ = sfm.run_sfm(p1_I=p1_I_keypoints, p2_I=p2_I_keypoints, K=K)
    _, best_inlier_mask, _ = pnp_ransac_localization_cv2(
        p_I_keypoints=p1_I_keypoints,
        p_W_landmarks=p_W,
        K=K,
    )

    return (
        Keypoints2D(p1_I_keypoints[:, best_inlier_mask]),
        Keypoints2D(p2_I_keypoints[:, best_inlier_mask]),
        Landmarks3D(p_W[:, best_inlier_mask]),
    )
