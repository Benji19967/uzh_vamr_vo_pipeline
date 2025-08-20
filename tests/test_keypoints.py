import numpy as np
import pytest

from src.features.keypoints import keep_unique


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
