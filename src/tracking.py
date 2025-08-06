import sys
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

import plot
from features.features_cv2 import good_features_to_track
from image import Image
from localization import ransacLocalization
from structure_from_motion import sfm
from utils.utils_cv2 import from_cv2, to_cv2

# TODO: add a note about notation / documentation regarding (x,y) vs (y,x)


def assert_dtype_int(arr: np.ndarray):
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("arr is not of type int")


def keep_unique(p_P: np.ndarray, p_P_existing: np.ndarray):
    """
    Remove existing points from p_P

    Args:
        - p_P           np.ndarray(2,N) | (x,y)
        - p_P_existing  np.ndarray(2,N) | (x,y)
    """
    assert_dtype_int(p_P)
    assert_dtype_int(p_P_existing)

    # Example arrays

    # Transpose to shape (N, 2) for easy comparison
    p_P = p_P.T  # shape (4, 2)
    p_P_existing = p_P_existing.T  # shape (2, 2)

    # Create a mask of which rows in p_P are NOT in p_P_existing
    mask = ~np.any(np.all(p_P[:, None] == p_P_existing[None, :], axis=2), axis=1)

    # Filter p_P with the mask, then transpose back to shape (2, K)
    p_P_filtered = p_P[mask].T

    return p_P_filtered


def run_klt(
    images: Sequence[Image],
    p_P_keypoints_initial: np.ndarray,
    p_W_landmarks_initial: np.ndarray,
    K: np.ndarray,
):
    """
    Run KLT on the images

    Args:
        - images list[np.ndarray]
        - p_P_keypoints_initial np.ndarray(2,N) | (x,y)
        - K np.ndarray(3, 3): camera matrix
    """
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # print(p_P_keypoints_initial)
    # print(p_W_landmarks_initial)

    p_P_candidate_keypoints_0 = np.zeros((2, 1))
    p_P_first_observations_0 = np.zeros((2, 1))
    p0_P_keypoints_cv2 = to_cv2(p_P_keypoints_initial)

    for i_0, i_1 in zip(images, images[1:]):
        # calculate optical flow
        p1_P_keypoints_cv2, st, err = cv2.calcOpticalFlowPyrLK(
            i_0.img, i_1.img, p0_P_keypoints_cv2, None, **lk_params
        )

        # Select good points
        p0_P_keypoints_cv2 = p0_P_keypoints_cv2[st == 1]
        p1_P_keypoints_cv2 = p1_P_keypoints_cv2[st == 1]

        p0_P_keypoints = from_cv2(p0_P_keypoints_cv2)
        R_C_W, t_C_W, best_inlier_mask, _, _ = ransacLocalization(
            p_P_keypoints=p0_P_keypoints,
            p_W_landmarks=p_W_landmarks_initial,
            K=K,
        )
        plot.plot_keypoints(img=i_0.img, p_P_keypoints=p0_P_keypoints)
        if R_C_W is not None:
            print(-R_C_W @ t_C_W)

        # print(i_0.img.shape)
        # plot.plot_tracking(
        #     I0_keypoints=from_cv2(p0_P_keypoints_cv2)[:, best_inlier_mask],
        #     I1_keypoints=from_cv2(p1_P_keypoints_cv2)[:, best_inlier_mask],
        #     figsize_pixels_x=i_0.img.shape[1],
        #     figsize_pixels_y=i_0.img.shape[0],
        # )

        # p0_P_keypoints_cv2 = p1_P_keypoints_cv2.reshape(-1, 1, 2)
        p_W_landmarks_initial = p_W_landmarks_initial[:, best_inlier_mask]
        p0_P_keypoints_cv2 = to_cv2(from_cv2(p1_P_keypoints_cv2)[:, best_inlier_mask])

        # Add new candidate keypoints
        p_P_candidate_keypoints_1 = good_features_to_track(
            img=i_1.img, max_features=200
        )
        p_P_candidate_keypoints_new = keep_unique(
            p_P=p_P_candidate_keypoints_1,
            p_P_existing=from_cv2(p0_P_keypoints_cv2).astype(np.int16),
        )
        p_P_candidate_keypoints_new = keep_unique(
            p_P=p_P_candidate_keypoints_1,
            p_P_existing=p_P_candidate_keypoints_0,
        )
        p_P_first_observations = p_P_candidate_keypoints_new

        # plot.plot_keypoints(
        #     img=i_1.img, p_P_keypoints=p_P_candidate_keypoints_new, fmt="gx"
        # )
