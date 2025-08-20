import numpy as np
import pytest

from src.features.keypoints import (
    find_keypoints,
    get_keypoint_correspondences,
    keep_unique,
)
from src.image import Image


def test_keep_unique_removes_existing_points():

    p_I = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )
    p_I_existing = np.array(
        [
            [2, 4],
            [6, 8],
        ]
    )

    result = keep_unique(p_I, p_I_existing)
    expected = np.array(
        [
            [1, 3],
            [5, 7],
        ]
    )

    assert np.array_equal(result, expected)


def test_keep_unique_no_existing_points():

    p_I = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    p_I_existing = np.empty((2, 0), dtype=np.int16)

    result = keep_unique(p_I, p_I_existing)
    assert np.array_equal(result, p_I)


def test_keep_unique_all_points_existing():

    p_I = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    p_I_existing = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    result = keep_unique(p_I, p_I_existing)
    assert result.shape == (2, 0)


def test_keep_unique_empty_input():

    p_I = np.empty((2, 0), dtype=np.int16)
    p_I_existing = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    result = keep_unique(p_I, p_I_existing)
    assert result.shape == (2, 0)


def test_keep_unique_raises_on_non_int_input():
    p_I = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    p_I_existing = np.array(
        [
            [1, 2],
            [3, 4],
        ],
    )
    with pytest.raises(TypeError):
        keep_unique(p_I, p_I_existing)

    p_I = np.array(
        [
            [1, 2],
            [3, 4],
        ],
    )
    p_I_existing = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(TypeError):
        keep_unique(p_I, p_I_existing)


def _same_columns(a, b):
    if a.shape != b.shape:
        return False

    a_cols = {tuple(a[:, i]) for i in range(a.shape[1])}
    b_cols = {tuple(b[:, i]) for i in range(b.shape[1])}

    return a_cols == b_cols


def test_find_keypoints_returns_expected_shape():
    img = np.zeros((100, 100), dtype=np.uint8)
    # Add some corners
    img[22, 22] = 255
    img[77, 77] = 255
    max_keypoints = 2

    p_I_new_keypoints, num_new_candidate_keypoints = find_keypoints(img, max_keypoints)
    expected = np.array(
        [
            [22, 77],
            [22, 77],
        ]
    )
    assert _same_columns(
        p_I_new_keypoints, expected
    )  # columns could be in different order
    assert num_new_candidate_keypoints == 2


def test_find_keypoints_excludes_existing_points():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[22, 22] = 255
    img[77, 77] = 255
    img[82, 31] = 255  # (y, x)
    max_keypoints = 3

    # Exclude one keypoint
    exclude = [np.array([[77], [77]])]  # (x, y)
    p_I_new_keypoints_filtered, num_new_candidate_keypoints = find_keypoints(
        img, max_keypoints, exclude=exclude
    )
    expected = np.array(
        [
            [22, 31],
            [22, 82],
        ]
    )
    assert _same_columns(
        p_I_new_keypoints_filtered, expected
    )  # columns could be in different order
    assert num_new_candidate_keypoints == 2


def test_find_keypoints_with_no_keypoints():
    img = np.zeros((10, 10), dtype=np.uint8)
    max_keypoints = 5
    p_I_new_keypoints, num_new_candidate_keypoints = find_keypoints(img, max_keypoints)
    assert p_I_new_keypoints.shape == (2, 0)
    assert num_new_candidate_keypoints == 0
