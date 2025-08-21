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


def test_to_hom_2d():
    p = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    result = points.to_hom(p)
    expected = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [1, 1, 1],
        ]
    )
    assert np.array_equal(result, expected)


def test_to_hom_3d():
    p = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    result = points.to_hom(p)
    expected = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [1, 1, 1],
        ]
    )
    assert np.array_equal(result, expected)


def test_compute_bearing_angles_with_translation():
    p_I_1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    p_I_2 = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
        ]
    )
    pose = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    poses_A = np.empty((12, 3))
    poses_A[:, 0] = pose.flatten()
    poses_A[:, 1] = pose.flatten()
    poses_A[:, 2] = pose.flatten()
    T_C_W = np.eye(3, 4)
    K = np.eye(3)

    angles_deg_expected = np.array([0.0, 54.74, 54.74])
    mask_angle_expected = np.array([False, True, True])

    _, angles_deg, mask_angle = points.compute_bearing_angles_with_translation(
        p_I_1=p_I_1, p_I_2=p_I_2, poses_A=poses_A, T_C_W=T_C_W, K=K, min_angle=5.0
    )
    assert np.allclose(angles_deg, angles_deg_expected, atol=1e-2)
    assert np.array_equal(mask_angle, mask_angle_expected)
