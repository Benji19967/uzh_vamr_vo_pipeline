from typing import TypeVar

import cv2
import numpy as np

from src.structures.keypoints2D import BaseKeypoints2D

T = TypeVar("T", bound=BaseKeypoints2D)

KLT_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  # type: ignore
)


def run_klt(
    image_0: np.ndarray, image_1: np.ndarray, keypoints: T
) -> tuple[T, np.ndarray]:
    """
    Run KLT on the images: track keypoints from image_0 to image_1

    Args:
        - image_0 np.ndarray
        - image_1 np.ndarray
        - p0_I_keypoints np.ndarray(2,N) | (x,y)

    Returns:
        - p0_I_keypoints np.ndarray(2,N) | (x,y)
        - status np.ndarray(N, ): True/False whether the point was successfully tracked
    """

    # calculate optical flow
    keypoints_cv2, status, err = cv2.calcOpticalFlowPyrLK(  # type: ignore
        image_0, image_1, keypoints.to_cv2(), None, **KLT_PARAMS
    )

    def from_cv2_status(st):
        """
        Returns:
            status: np.ndarray(N,): 1 if point tracked else 0
        """
        return st.T[0]

    keypoints = keypoints.__class__.from_cv2(keypoints_cv2)
    if keypoints_cv2 is None:
        return (
            keypoints,
            np.full((0), False),
        )
    return (
        keypoints,
        from_cv2_status(st=status).astype(np.bool8),
    )
