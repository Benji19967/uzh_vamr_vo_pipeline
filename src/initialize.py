import sys
import time

import numpy as np

from features.features import Descriptors, HarrisScores, Keypoints
from image import Image
from localization import ransacLocalization
from structure_from_motion import sfm

NUM_KEYPOINTS = 200


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
    kps = []
    for image in [I_0, I_1]:
        hs = HarrisScores(image=image)
        kp = Keypoints(image=image, scores=hs.scores)
        kp.select(num_keypoints=NUM_KEYPOINTS)
        keypoints.append(kp.keypoints)
        kps.append(kp)
        # kp.plot()
        # print(kp.keypoints)
        desc = Descriptors(image=image, keypoints=kp.keypoints)
        # desc.plot()
        descriptors.append(desc.descriptors)

    matches = Descriptors.match(
        query_descriptors=descriptors[1], db_descriptors=descriptors[0]
    )
    # Descriptors.plot_matches(
    #     matches=matches, query_keypoints=keypoints[1], database_keypoints=keypoints[0]
    # )

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

    # kps[0].plot(I_0_matched_keypoints)
    # kps[1].plot(I_1_matched_keypoints)

    # Switch pixel coordinates from (y, x) to (x, y)
    I_0_matched_keypoints[[0, 1], :] = I_0_matched_keypoints[[1, 0], :]
    I_1_matched_keypoints[[0, 1], :] = I_1_matched_keypoints[[1, 0], :]

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
    print(p1_P_keypoints.shape)
    # sys.exit()

    p_W, _, _ = sfm.run_sfm(p1_P=p1_P_keypoints, p2_P=p2_P_keypoints, K=K)
    R_C_W, t_C_W, best_inlier_mask, _, _ = ransacLocalization(
        p_P_keypoints=p1_P_keypoints,
        p_W_landmarks=p_W,
        K=K,
    )
    # print(best_inlier_mask)

    return (
        p1_P_keypoints[:, best_inlier_mask],
        p2_P_keypoints[:, best_inlier_mask],
        p_W[:, best_inlier_mask],
    )
