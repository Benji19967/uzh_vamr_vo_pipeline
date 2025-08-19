import numpy as np

from src.structure_from_motion.linear_triangulation import (
    linear_triangulation,
    reprojection_error,
)


def test_linear_triangulation_multiple_points():
    # Two points in two images
    p1_I_hom = np.array(
        [
            [0.5, 0.6],
            [0.75, 0.4],
            [1.0, 1.0],
        ]
    )
    p2_I_hom = np.array(
        [
            [0.25, 0.4],
            [0.75, 0.4],
            [1.0, 1.0],
        ]
    )

    M1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    M2 = np.array(
        [
            [1, 0, 0, -1],  # translation in x
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    p_W_hom_expected = np.array(
        [
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 5.0],
            [1.0, 1.0],
        ]
    )

    p_W_hom = linear_triangulation(p1_I_hom, p2_I_hom, M1, M2)

    assert p_W_hom.shape == (4, 2)
    np.testing.assert_array_almost_equal(p_W_hom, p_W_hom_expected)


def test_reprojection_error_zero_error():
    # 3D points
    p_W_hom = np.array(
        [
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 5.0],
            [1.0, 1.0],
        ]
    )
    K = np.eye(3)
    T_C_W_1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    T_C_W_2 = np.array(
        [
            [1, 0, 0, -1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    p_I_1 = np.array(
        [
            [0.5, 0.6],
            [0.75, 0.4],
        ]
    )
    p_I_2 = np.array(
        [
            [0.25, 0.4],
            [0.75, 0.4],
        ]
    )

    error1 = reprojection_error(p_W_hom, p_I_1, T_C_W_1, K)
    assert np.isclose(error1, 0.0)
    error2 = reprojection_error(p_W_hom, p_I_2, T_C_W_2, K)
    assert np.isclose(error2, 0.0)

    # Wrong camera poses
    error3 = reprojection_error(p_W_hom, p_I_1, T_C_W_2, K)
    assert not np.isclose(error3, 0.0)
    error4 = reprojection_error(p_W_hom, p_I_2, T_C_W_1, K)
    assert not np.isclose(error4, 0.0)
