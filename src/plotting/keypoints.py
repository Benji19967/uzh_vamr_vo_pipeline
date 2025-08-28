from typing import Any

import cv2
import numpy as np

cv2: Any


def plot_keypoints(
    img: np.ndarray,
    p_I_keypoints: np.ndarray | list[np.ndarray],
    marker_size: int | list[int] = 5,
    color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
    thickness: int | list[int] = 1,
) -> None:
    """
    Significantly faster than plotting with matplotlib

    Args:
        img (np.ndarray): Image to draw on.
        p_I_keypoints (np.ndarray(2, N) or list[np.ndarray(2, N)]): Keypoints to draw
        marker_size (int or list[int]): Size of the markers.
        color (tuple[int, int, int] or list[tuple[int, int, int]]): Color of the markers in BGR format.
        thickness (int or list[int]): Thickness of the markers.
    """
    img_out = draw_keypoints(img, p_I_keypoints, marker_size, color, thickness)

    cv2.imshow("Keypoints", img_out)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def draw_keypoints(
    img: np.ndarray,
    p_I_keypoints: np.ndarray | list[np.ndarray],
    marker_size: int | list[int] = 5,
    color: tuple[int, int, int] | list[tuple[int, int, int]] = (255, 0, 0),
    thickness: int | list[int] = 1,
):
    """
    Creates a cv2 image and draws keypoints on it.

    Args:
        img (np.ndarray): Image to draw on.
        p_I_keypoints (np.ndarray(2, N) or list[np.ndarray(2, N)]): Keypoints to draw
        marker_size (int or list[int]): Size of the markers.
        color (tuple[int, int, int] or list[tuple[int, int, int]]): Color of the markers in BGR format.
        thickness (int or list[int]): Thickness of the markers.

    Returns:
        np.ndarray: Image with drawn keypoints.
    """
    img_out = img.copy()
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    def draw_marker(p_I_keypoints, s, c, t):
        for x, y in p_I_keypoints.T:
            cv2.drawMarker(
                img_out,
                (int(x), int(y)),
                c,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=s,
                thickness=t,
                line_type=cv2.LINE_AA,
            )

    if isinstance(p_I_keypoints, list):
        assert isinstance(marker_size, list)
        assert isinstance(color, list)
        assert isinstance(thickness, list)
        assert len(p_I_keypoints) == len(marker_size) == len(color) == len(thickness)

        for p_I_kps, s, c, t in zip(p_I_keypoints, marker_size, color, thickness):
            draw_marker(p_I_kps, s, c, t)
    else:
        draw_marker(p_I_keypoints, marker_size, color, thickness)
    return img_out
