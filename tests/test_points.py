import numpy as np

from src.utils import points


def test_apply_mask_basic():
    points_arr = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([True, False, True])
    expected = points_arr[:, mask]
    result = points.apply_mask(points_arr, mask)
    assert np.array_equal(result, expected)


def test_apply_mask_all_true():
    points_arr = np.array([[1, 2], [3, 4]])
    mask = np.array([True, True])
    expected = points_arr[:, mask]
    result = points.apply_mask(points_arr, mask)
    assert np.array_equal(result, expected)


def test_apply_mask_all_false():
    points_arr = np.array([[1, 2], [3, 4]])
    mask = np.array([False, False])
    expected = points_arr[:, mask]
    result = points.apply_mask(points_arr, mask)
    assert np.array_equal(result, expected)


def test_apply_mask_empty_points():
    points_arr = np.array([])
    mask = np.array([True, False])
    result = points.apply_mask(points_arr, mask)
    assert np.array_equal(result, points_arr)


def test_apply_mask_many_basic():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[7, 8, 9], [10, 11, 12]])
    mask = np.array([True, False, True])
    arr1_masked, arr2_masked = points.apply_mask_many([arr1, arr2], mask)
    assert np.array_equal(arr1_masked, arr1[:, mask])
    assert np.array_equal(arr2_masked, arr2[:, mask])


def test_apply_mask_many_empty_array():
    arr1 = np.array([])
    arr2 = np.array([[1, 2], [3, 4]])
    mask = np.array([True, False])
    arr1_masked, arr2_masked = points.apply_mask_many([arr1, arr2], mask)
    assert np.array_equal(arr1_masked, arr1)
    assert np.array_equal(arr2_masked, arr2[:, mask])


def test_apply_mask_many_all_empty():
    arr1 = np.array([])
    arr2 = np.array([])
    mask = np.array([True, False])
    arr1_masked, arr2_masked = points.apply_mask_many([arr1, arr2], mask)
    assert np.array_equal(arr1_masked, arr1)
    assert np.array_equal(arr2_masked, arr2)


def test_to_cv2_basic():
    p_I = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    result = points.to_cv2(p_I)
    expected = np.array(
        [
            [[1.0, 4.0]],
            [[2.0, 5.0]],
            [[3.0, 6.0]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)
    assert result.dtype == np.float32
    assert result.shape == (3, 1, 2)


def test_to_cv2_empty():
    p_I = np.empty((2, 0))
    result = points.to_cv2(p_I)
    expected = np.empty((0, 1, 2), dtype=np.float32)
    assert np.array_equal(result, expected)
    assert result.shape == (0, 1, 2)


def test_to_cv2_dtype_conversion():
    p_I = np.array([[1, 2], [3, 4]], dtype=np.int64)
    result = points.to_cv2(p_I)
    assert result.dtype == np.float32


def test_from_cv2_basic():
    p_I = np.array(
        [
            [[1.0, 4.0]],
            [[2.0, 5.0]],
            [[3.0, 6.0]],
        ],
        dtype=np.float32,
    )
    result = points.from_cv2(p_I)
    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)
    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_from_cv2_empty():
    p_I = np.empty((0, 1, 2), dtype=np.float32)
    result = points.from_cv2(p_I)
    expected = np.empty((2, 0), dtype=np.float32)
    assert np.array_equal(result, expected)
    assert result.shape == (2, 0)
