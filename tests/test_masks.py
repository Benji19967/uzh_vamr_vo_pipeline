import numpy as np

from src.utils.masks import compose_masks


def test_compose_masks():
    mask1 = np.array([0, 1, 0, 1, 0, 1, 1])
    mask2 = np.array([0, 1, 1, 1])
    expected_result = np.array([0, 0, 0, 1, 0, 1, 1])

    result = compose_masks(mask1, mask2)

    assert np.array_equal(result, expected_result)
