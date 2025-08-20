import numpy as np

from src.transformations.transformations import world_to_camera


def test_world_to_camera_identity_transformation():
    p_W_hom = np.array(
        [
            [1, 9, 2],
            [3, 0, 3],
            [7, 1, 0],
            [1, 1, 1],
        ]
    )
    T_C_W = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    assert np.array_equal(world_to_camera(p_W_hom, T_C_W), p_W_hom[:3, :])


def test_world_to_camera_translation():
    p_W_hom = np.array(
        [
            [1, 9, 2],
            [3, 0, 3],
            [7, 1, 0],
            [1, 1, 1],
        ]
    )
    T_C_W = np.array(
        [
            [1, 0, 0, 3],
            [0, 1, 0, 2],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    p_W_expected = np.array(
        [
            [4, 12, 5],
            [5, 2, 5],
            [8, 2, 1],
        ]
    )
    assert np.array_equal(world_to_camera(p_W_hom, T_C_W), p_W_expected)


def test_world_to_camera_scaling():
    p_W_hom = np.array(
        [
            [1, 9, 2],
            [3, 0, 3],
            [7, 1, 0],
            [1, 1, 1],
        ]
    )
    T_C_W = np.array(
        [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 1],
        ]
    )
    p_W_expected = np.array(
        [
            [1, 9, 2],
            [6, 0, 6],
            [21, 3, 0],
        ]
    )
    assert np.array_equal(world_to_camera(p_W_hom, T_C_W), p_W_expected)
