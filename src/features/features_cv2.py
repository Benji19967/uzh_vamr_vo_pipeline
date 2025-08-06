import cv2 as cv
import numpy as np

from utils.utils_cv2 import from_cv2


def good_features_to_track(img: np.ndarray, max_features: int):
    """Shi-Tomasi Corner Detector

    Args:
     - img          np.ndarray: img to detect features from
     - max_features int: max number of features that will be detected

    Returns:
     - p_P_corners   np.ndarray(2,N): (x,y) coordinates of 2D corners/features detected
    """
    R_min = 0.01
    min_euclidean_distance_between_corners = 10
    corners = cv.goodFeaturesToTrack(
        img, max_features, R_min, min_euclidean_distance_between_corners
    )
    corners = np.int0(corners)
    p_P_corners = from_cv2(corners)  # type: ignore

    return p_P_corners
