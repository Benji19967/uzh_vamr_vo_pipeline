import cv2 as cv
import numpy as np

from src.utils.points import from_cv2

MIN_EUCLIDEAN_DISTANCE_BETWEEN_CORNERS = 20


def good_features_grid(
    img: np.ndarray,
    max_features: int,
    R_min: float = 0.01,
    min_distance: int = 5,
    grid_rows: int = 6,
    grid_cols: int = 8,
    max_features_per_cell: int = 10,
):
    """Shi-Tomasi Corner Detector -- find features in a grid
    for a better distribution of keypoints.

    Args:
     - img          np.ndarray: img to detect features from
     - max_features int: max number of features that will be detected

    Returns:
     - p_I_corners  np.ndarray(2,N): (x,y) coordinates of 2D corners/features detected
    """
    max_features_per_cell = max(
        max_features_per_cell, max_features // (grid_rows * grid_cols)
    )
    h, w = img.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    features = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Define cell region
            x_start = col * cell_w
            y_start = row * cell_h
            x_end = x_start + cell_w
            y_end = y_start + cell_h

            cell_img = img[y_start:y_end, x_start:x_end]

            # Detect features in the cell
            corners = cv.goodFeaturesToTrack(  # type: ignore
                cell_img,
                maxCorners=max_features_per_cell,
                qualityLevel=R_min,
                minDistance=min_distance,
            )

            if corners is not None:
                # Adjust coordinates to the full image
                for c in corners:
                    c_full = c[0] + np.array([x_start, y_start])
                    features.append(c_full)

    if not features:
        return np.zeros((2, 0), dtype=np.int16)

    # Convert to required format
    features = np.array(features, dtype=np.int0).T

    assert features.shape[1] <= max_features

    return features


def good_features_to_track(img: np.ndarray, max_features: int, R_min: float = 0.01):
    """Shi-Tomasi Corner Detector

    Args:
     - img          np.ndarray: img to detect features from
     - max_features int: max number of features that will be detected

    Returns:
     - p_I_corners   np.ndarray(2,N): (x,y) coordinates of 2D corners/features detected
    """
    corners = cv.goodFeaturesToTrack(  # type: ignore
        img, max_features, R_min, MIN_EUCLIDEAN_DISTANCE_BETWEEN_CORNERS
    )
    if corners is None:
        return np.zeros((2, 0), dtype=np.int16)

    corners = np.int0(corners)
    p_I_corners = from_cv2(corners)  # type: ignore

    return p_I_corners
