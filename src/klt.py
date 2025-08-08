import cv2
import numpy as np

from src.image import Image
from utils.utils_cv2 import from_cv2, to_cv2

KLT_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def run_klt(image_0: Image, image_1: Image, p0_I_keypoints: np.ndarray):
    """
    Run KLT on the images: track keypoints from image_0 to image_1

    Args:
        - image_0 Image
        - image_1 Image
        - p0_I_keypoints np.ndarray(2,N) | (x,y)

    Returns:
        - p0_I_keypoints np.ndarray(2,N) | (x,y)
        - status np.ndarray(N, ): True/False whether the point was successfully tracked
    """

    # calculate optical flow
    p0_I_keypoints_cv2 = to_cv2(p0_I_keypoints)
    p1_I_keypoints_cv2, status, err = cv2.calcOpticalFlowPyrLK(
        image_0.img, image_1.img, p0_I_keypoints_cv2, None, **KLT_PARAMS
    )

    def from_cv2_status(st):
        """
        Returns:
            status: np.ndarray(N,): 1 if point tracked else 0
        """
        return st.T[0]

    if p1_I_keypoints_cv2 is None:
        print("BAD")
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.full((0), False),
        )
    print("GOOD")
    return (
        from_cv2(p1_I_keypoints_cv2),
        from_cv2_status(st=status).astype(np.bool8),
    )
