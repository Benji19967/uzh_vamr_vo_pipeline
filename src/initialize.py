import sys
import time

import numpy as np

from features.features import Descriptors, HarrisScores, Keypoints
from image import Image
from localization import ransacLocalization
from src.features.features_cv2 import good_features_to_track
from structure_from_motion import sfm

NUM_KEYPOINTS = 1000


def get_keypoint_correspondences(
    I_0: Image, I_1: Image
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tuple[np.ndarray, np.ndarray]:
            (
                (2, N) matched_keypoints1 in pixel coordinates (x, y),
                (2, N) matched_keypoints2 in pixel coordinates (x, y)
            )
    """
    keypoints = []
    descriptors = []
    for image in [I_0, I_1]:
        p_P_corners = good_features_to_track(img=image.img, max_features=NUM_KEYPOINTS)
        keypoints.append(p_P_corners)

        desc = Descriptors(image=image, keypoints=p_P_corners)
        descriptors.append(desc.descriptors)

    matches = Descriptors.match(
        query_descriptors=descriptors[1], db_descriptors=descriptors[0]
    )

    I_0_keypoints = keypoints[0]
    I_1_keypoints = keypoints[1]
    I_1_indices = np.nonzero(matches >= 0)[0]
    I_0_indices = matches[I_1_indices]

    I_0_matched_keypoints = np.zeros((2, len(I_1_indices)))
    I_1_matched_keypoints = np.zeros((2, len(I_1_indices)))

    I_0_matched_keypoints[0:] = I_0_keypoints[0, I_0_indices]
    I_0_matched_keypoints[1:] = I_0_keypoints[1, I_0_indices]
    I_1_matched_keypoints[0:] = I_1_keypoints[0, I_1_indices]
    I_1_matched_keypoints[1:] = I_1_keypoints[1, I_1_indices]

    return I_0_matched_keypoints, I_1_matched_keypoints


def initialize(
    I_0: Image, I_1: Image, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        a (int): _description_
        b (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (
                (2xN) keypoints of image 1 in pixel coordinates (x, y),
                (2xN) keypoints of image 2 in pixel coordinates (x, y),
                (3xN) landmarks P_W in 3D coordinates (x, y, z)
            )
    """
    p1_P_keypoints, p2_P_keypoints = get_keypoint_correspondences(I_0=I_0, I_1=I_1)
    # print(p1_P_keypoints.shape)
    # sys.exit()

    p_W, _, _ = sfm.run_sfm(p1_P=p1_P_keypoints, p2_P=p2_P_keypoints, K=K)
    _, _, best_inlier_mask, _, _ = ransacLocalization(
        p_P_keypoints=p1_P_keypoints,
        p_W_landmarks=p_W,
        K=K,
    )
    # print(best_inlier_mask)
    # sys.exit()

    return (
        p1_P_keypoints[:, best_inlier_mask],
        p2_P_keypoints[:, best_inlier_mask],
        p_W[:, best_inlier_mask],
    )
