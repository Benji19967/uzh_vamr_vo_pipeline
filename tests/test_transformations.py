import numpy as np

from src.transformations import world_to_camera


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
