from typing import Sequence

import cv2
import numpy as np

import plot
from image import Image

# TODO: add a note about notation / documentation regarding (x,y) vs (y,x)


def to_cv2(p_P: np.ndarray) -> np.ndarray:
    """
    Convert format of points to match what cv2 expects

    Args:
        p_P (np.ndarray): (2xN)

    Returns:
        np.ndarray: (Nx1x2)
    """
    p_P = p_P.astype(np.float32)
    N = p_P.shape[1]
    return p_P.T.reshape((N, 1, 2))


def from_cv2(p_P: np.ndarray) -> np.ndarray:
    """
    Convert format of points from cv2 format to (2xN)

    Args:
        p_P (np.ndarray): (Nx1x2)

    Returns:
        np.ndarray: (2xN)
    """
    return p_P.T.reshape(2, -1)


def run_klt(images: Sequence[Image], p_P_keypoints_initial: np.ndarray):
    """
    Run KLT on the images

    Args:
        images (list[np.ndarray])
        p_P_keypoints_initial (2xN) | (x,y)
    """
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    p0_P_keypoints_cv2 = to_cv2(p_P_keypoints_initial)
    for i_0, i_1 in zip(images, images[1:]):
        # calculate optical flow
        p1_P_keypoints_cv2, st, err = cv2.calcOpticalFlowPyrLK(
            i_0.img, i_1.img, p0_P_keypoints_cv2, None, **lk_params
        )

        # Select good points
        p0_P_keypoints_cv2 = p0_P_keypoints_cv2[st == 1]
        p1_P_keypoints_cv2 = p1_P_keypoints_cv2[st == 1]

        plot.plot_tracking(
            I0_keypoints=from_cv2(p0_P_keypoints_cv2),
            I1_keypoints=from_cv2(p1_P_keypoints_cv2),
        )

        p0_P_keypoints_cv2 = p1_P_keypoints_cv2.reshape(-1, 1, 2)
