from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

cv2: Any

DPI = 227  # Mac M1


def plot_tracking_cv2(
    I0_keypoints: np.ndarray, I1_keypoints: np.ndarray, shape, winname="Tracks"
):
    """
    Draws tracking lines on a white canvas using image1 coordinates.

    Args:
        I0_keypoints (np.ndarray): (2, N) array of keypoints in image 0 (x, y).
        I1_keypoints (np.ndarray): (2, N) array of keypoints in image 1 (x, y).
        shape (tuple): (H, W) shape of the image/canvas.
        winname (str): Window name for cv2.imshow.
    """
    canvas = draw_tracks(I0_keypoints, I1_keypoints, shape)

    cv2.imshow(winname, canvas)
    cv2.waitKey(10)
    cv2.destroyAllWindows()


def draw_tracks(I0_keypoints, I1_keypoints, shape):
    """
    Draws tracking lines on a white canvas using image1 coordinates.

    Args:
        I0_keypoints (np.ndarray): (2, N) array of keypoints in image 0 (x, y).
        I1_keypoints (np.ndarray): (2, N) array of keypoints in image 1 (x, y).
        shape (tuple): (H, W) shape of the image/canvas.
        winname (str): Window name for cv2.imshow.
    """
    assert (
        I0_keypoints.shape[0] == 2 and I1_keypoints.shape[0] == 2
    ), "Keypoints must be shape (2, N)"
    N = I0_keypoints.shape[1]
    assert I1_keypoints.shape[1] == N, "Both keypoint arrays must have same N"

    H, W = shape
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white canvas

    for i in range(N):
        x0, y0 = I0_keypoints[:, i]
        x1, y1 = I1_keypoints[:, i]
        color = (0, 255, 0)
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1), int(y1))
        cv2.arrowedLine(canvas, pt1, pt0, color, 1, tipLength=0.2)

    return canvas
